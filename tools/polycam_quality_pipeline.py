#!/usr/bin/env python3
import csv
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


GLCAM_IN_CVCAM = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)

QUALITY_WEIGHTS = {
    "pose_alignment": 0.30,
    "depth_quality": 0.25,
    "mask_quality": 0.20,
    "object_visibility": 0.10,
    "rgb_sharpness": 0.10,
    "intrinsics_stability": 0.05,
}


@dataclass
class QualityRecord:
    timestamp: str
    sharpness: float = 0.0
    exposure_score: float = 1.0
    mean_luminance: float = 0.0
    dark_pixel_ratio: float = 0.0
    bright_pixel_ratio: float = 0.0
    mask_ratio: float = 0.0
    mask_border_contact: float = 0.0
    mask_depth_valid_ratio: float = 0.0
    depth_valid_ratio: float = 0.0
    high_confidence_ratio: float = 0.0
    medium_confidence_ratio: float = 0.0
    low_confidence_ratio: float = 0.0
    intrinsics_deviation: float = 0.0
    pose_translation_delta: float = 0.0
    pose_rotation_delta: float = 0.0
    pose_alignment_error: float = math.nan
    pose_alignment_p75: float = math.nan
    pose_alignment_p90: float = math.nan
    pose_alignment_overlap_ratio: float = 0.0
    view_redundancy: float = 0.0
    quality_score: float = 0.0
    accepted: bool = True
    reject_reasons: list[str] = field(default_factory=list)
    selected: bool = False


@dataclass
class CandidateFrame:
    timestamp: str
    rgb_path: Path
    depth_path: Path
    confidence_path: Path
    camera_path: Path
    K_raw: np.ndarray
    K: np.ndarray
    polycam_cam_in_world: np.ndarray
    cam_in_ob: np.ndarray
    rgb_small: np.ndarray | None = None
    depth_mm: np.ndarray | None = None
    confidence: np.ndarray | None = None
    mask_original: np.ndarray | None = None
    mask_depth_safe: np.ndarray | None = None
    points_ob: np.ndarray | None = None
    quality: QualityRecord | None = None
    output_index: int | None = None


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def list_by_stem(directory, suffixes=None):
    out = {}
    duplicates = []
    if not directory.is_dir():
        return out, duplicates
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        if p.stem in out:
            duplicates.append(p.stem)
        out[p.stem] = p
    return out, sorted(set(duplicates))


def timestamp_key(value):
    return int(value) if str(value).isdigit() else str(value)


def read_camera(path):
    data = read_json(path)
    K = np.array(
        [[data["fx"], 0.0, data["cx"]], [0.0, data["fy"], data["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    for r in range(3):
        for c in range(4):
            T[r, c] = float(data[f"t_{r}{c}"])
    return data, K, T


def validate_K(K, width, height):
    return (
        K.shape == (3, 3)
        and np.all(np.isfinite(K))
        and K[0, 0] > 0
        and K[1, 1] > 0
        and 0 <= K[0, 2] < width
        and 0 <= K[1, 2] < height
    )


def rotation_angle(R1, R2):
    rel = R1.T @ R2
    val = (np.trace(rel) - 1.0) * 0.5
    return float(math.acos(float(np.clip(val, -1.0, 1.0))))


def validate_pose(T):
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5):
        return False
    R = T[:3, :3]
    return np.allclose(R.T @ R, np.eye(3), atol=2e-3) and abs(np.linalg.det(R) - 1.0) <= 2e-3


def robust_mad(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


def sharpness_score(rgb_bgr):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(rgb_bgr):
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    dark = float((gray < 8).mean())
    bright = float((gray > 247).mean())
    severe = dark > 0.70 or bright > 0.45 or mean < 18 or mean > 238
    penalty = max(dark / 0.70, bright / 0.45, max(0.0, 18 - mean) / 18.0, max(0.0, mean - 238) / 17.0)
    return mean, dark, bright, max(0.0, 1.0 - min(1.0, penalty)), severe


def confidence_filter(depth, conf, mode):
    if conf is None or conf.shape != depth.shape:
        raise RuntimeError(f"confidence shape {None if conf is None else conf.shape} does not match depth {depth.shape}")
    high_value = int(conf.max())
    high = conf == high_value if high_value > 0 else np.zeros_like(conf, dtype=bool)
    medium = (conf > 0) & ~high
    low = conf == 0
    if mode == "all":
        keep = np.ones(conf.shape, dtype=bool)
    elif mode == "high":
        keep = high
    else:
        keep = high | medium
    filtered = depth.copy()
    filtered[~keep] = 0
    return filtered, high, medium, low


def remove_small_components(mask, min_area=16, keep_area_ratio=0.08):
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = float(areas.max()) if areas.size else 0.0
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area and area >= largest * keep_area_ratio:
            keep |= labels == label
    return keep.astype(np.uint8) * 255


def mask_border_contact(mask):
    m = mask > 0
    if m.sum() == 0:
        return 0.0
    border = np.zeros_like(m)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return float((m & border).sum() / max(1, m.sum()))


def make_depth_safe_mask(mask):
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) < 80:
        return m * 255
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    if eroded.sum() < max(32, 0.45 * m.sum()):
        return m * 255
    return eroded.astype(np.uint8) * 255


def depth_to_points(depth_mm, mask, K, max_points=2500, stride=1):
    depth_m = depth_mm.astype(np.float64) / 1000.0
    valid = (depth_m > 0.001) & (mask > 0)
    if stride > 1:
        sample = np.zeros_like(valid)
        sample[::stride, ::stride] = True
        valid &= sample
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(xs) > max_points:
        ids = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[ids], ys[ids]
    z = depth_m[ys, xs]
    x = (xs.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (ys.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def transform_points(T, pts):
    if len(pts) == 0:
        return pts
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
    return (T @ pts_h.T).T[:, :3]


def write_ply(path, point_sets):
    pts_all = []
    color_all = []
    palette = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
        ],
        dtype=np.uint8,
    )
    for i, pts in enumerate(point_sets):
        if pts is None or len(pts) == 0:
            continue
        pts_all.append(pts)
        color_all.append(np.tile(palette[i % len(palette)], (len(pts), 1)))
    if not pts_all:
        return
    pts = np.concatenate(pts_all, axis=0)
    colors = np.concatenate(color_all, axis=0)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def load_yolo_model(model_path):
    from ultralytics import YOLO

    try:
        return YOLO(model_path)
    except Exception as exc:
        try:
            import torch
            from ultralytics.nn.tasks import SegmentationModel

            torch.serialization.add_safe_globals([SegmentationModel])
            return YOLO(model_path)
        except Exception:
            pass
        try:
            import torch

            original_load = torch.load

            def torch_load_compat(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = torch_load_compat
            try:
                return YOLO(model_path)
            finally:
                torch.load = original_load
        except Exception:
            raise exc


def yolo_class_id(model, class_name):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        iterable = names.items()
    else:
        iterable = enumerate(names or [])
    for idx, name in iterable:
        if str(name).lower() == class_name.lower():
            return int(idx)
    raise RuntimeError(f"YOLO class '{class_name}' was not found in model names: {names}")


def yolo_segmentation_mask(model, class_id, rgb_bgr, target_shape, conf=0.25, iou=0.7, device=None):
    results = model.predict(rgb_bgr[..., ::-1], conf=conf, iou=iou, device=device, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no result")
    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise RuntimeError("YOLO result has no segmentation masks")
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(cls))
    masks = result.masks.data.detach().cpu().numpy()
    H, W = target_shape[:2]
    candidates = []
    for i, cid in enumerate(cls):
        if cid != class_id:
            continue
        mask = cv2.resize(masks[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        center_score = 1.0 - min(1.0, np.linalg.norm(np.array([xs.mean() / W, ys.mean() / H]) - 0.5) / 0.7)
        area_score = min(float(mask.mean()) / 0.20, 1.0)
        candidates.append((float(confs[i]) + 0.25 * center_score + 0.15 * area_score, mask))
    if not candidates:
        raise RuntimeError(f"YOLO found masks, but none for class id {class_id}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].astype(np.uint8) * 255


def auto_depth_mask(depth_mm):
    valid = (depth_mm > 0).astype(np.uint8)
    if valid.sum() == 0:
        return valid
    ys, xs = np.where(valid)
    z = depth_mm[ys, xs].astype(np.float64)
    med, mad = robust_mad(z)
    lo = med - 4.0 * max(mad, 1.0)
    hi = med + 4.0 * max(mad, 1.0)
    mask = valid & (depth_mm >= lo) & (depth_mm <= hi)
    return remove_small_components(mask.astype(np.uint8) * 255, min_area=64)


def make_contactsheet(frames, out_file, label_func, max_items=None, tile_w=224):
    chosen = frames[:max_items] if max_items else frames
    if not chosen:
        return
    tiles = []
    for fr in chosen:
        img = fr.rgb_small.copy()
        if fr.mask_original is not None:
            red = np.zeros_like(img)
            red[:, :, 2] = 255
            img = np.where(fr.mask_original[..., None] > 0, cv2.addWeighted(img, 0.55, red, 0.45, 0), img)
        h, w = img.shape[:2]
        tile_h = max(1, int(h * tile_w / w))
        tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        for j, line in enumerate(label_func(fr).split("\n")[:5]):
            cv2.putText(tile, line, (6, 18 + j * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cols = min(6, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    th, tw = tiles[0].shape[:2]
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th : r * th + th, c * tw : c * tw + tw] = tile
    cv2.imwrite(str(out_file), sheet)


def compute_pose_alignment(frames, diag, name, iterations=2, pose_mad_factor=3.0, max_pose_error_mm=None):
    usable = [fr for fr in frames if fr.quality.accepted and fr.points_ob is not None and len(fr.points_ob) > 20]
    write_ply(diag / f"pose_alignment_{name}.ply", [fr.points_ob for fr in usable])
    if cKDTree is None or len(usable) < 3:
        return {"median": math.nan, "p90": math.nan, "worst": math.nan, "pass": len(usable) >= 3}

    current = usable[:]
    stats = {}
    for _ in range(max(1, iterations)):
        all_pts = np.concatenate([fr.points_ob for fr in current if len(fr.points_ob) > 0], axis=0)
        if len(all_pts) > 60000:
            all_pts = all_pts[np.linspace(0, len(all_pts) - 1, 60000).astype(int)]
        errors = []
        for fr in current:
            ref_pts = np.concatenate([other.points_ob for other in current if other is not fr and len(other.points_ob) > 0], axis=0)
            if len(ref_pts) > 50000:
                ref_pts = ref_pts[np.linspace(0, len(ref_pts) - 1, 50000).astype(int)]
            if len(ref_pts) < 50 or len(fr.points_ob) < 20:
                continue
            tree = cKDTree(ref_pts)
            dists, _ = tree.query(fr.points_ob, k=1, workers=-1)
            q = fr.quality
            q.pose_alignment_error = float(np.median(dists))
            q.pose_alignment_p75 = float(np.percentile(dists, 75))
            q.pose_alignment_p90 = float(np.percentile(dists, 90))
            q.pose_alignment_overlap_ratio = float((dists < max(0.01, 2.5 * q.pose_alignment_error)).mean())
            errors.append(q.pose_alignment_error)
        med, mad = robust_mad(errors)
        if not np.isfinite(med):
            break
        threshold = med + pose_mad_factor * max(mad, 1e-6)
        if max_pose_error_mm is not None:
            threshold = min(threshold, max_pose_error_mm / 1000.0)
        next_current = []
        for fr in current:
            err = fr.quality.pose_alignment_error
            if np.isfinite(err) and err > threshold:
                if "pose_alignment_outlier" not in fr.quality.reject_reasons:
                    fr.quality.reject_reasons.append("pose_alignment_outlier")
                fr.quality.accepted = False
            else:
                next_current.append(fr)
        current = next_current
        stats = {"median": med, "mad": mad, "threshold": threshold}

    filtered = [fr for fr in frames if fr.quality.accepted and fr.points_ob is not None and len(fr.points_ob) > 20]
    write_ply(diag / f"pose_alignment_{name}_filtered.ply", [fr.points_ob for fr in filtered])
    values = np.array([fr.quality.pose_alignment_error for fr in filtered], dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "median": float(np.median(values)) if values.size else math.nan,
        "p90": float(np.percentile(values, 90)) if values.size else math.nan,
        "worst": float(values.max()) if values.size else math.nan,
        "threshold": stats.get("threshold", math.nan),
        "pass": len(filtered) >= 3,
    }


def compute_quality_scores(frames, sharpness_floor):
    accepted = [fr for fr in frames if fr.quality.accepted]
    align_vals = np.array([fr.quality.pose_alignment_error for fr in accepted], dtype=np.float64)
    align_vals = align_vals[np.isfinite(align_vals)]
    align_ref = float(np.percentile(align_vals, 75)) if align_vals.size else 0.02
    for fr in frames:
        q = fr.quality
        if not q.accepted:
            q.quality_score = 0.0
            continue
        pose_score = 1.0 if not np.isfinite(q.pose_alignment_error) else 1.0 - min(1.0, q.pose_alignment_error / max(align_ref * 2.0, 1e-6))
        depth_score = min(1.0, q.mask_depth_valid_ratio / 0.85)
        mask_score = min(1.0, q.mask_ratio / 0.08) * (1.0 - min(1.0, q.mask_border_contact * 8.0))
        visibility_score = min(1.0, q.mask_ratio / 0.18)
        sharp_score = min(1.0, q.sharpness / max(sharpness_floor * 2.0, 1.0))
        intr_score = 1.0 - min(1.0, q.intrinsics_deviation / 0.02)
        q.quality_score = (
            QUALITY_WEIGHTS["pose_alignment"] * pose_score
            + QUALITY_WEIGHTS["depth_quality"] * depth_score
            + QUALITY_WEIGHTS["mask_quality"] * mask_score
            + QUALITY_WEIGHTS["object_visibility"] * visibility_score
            + QUALITY_WEIGHTS["rgb_sharpness"] * sharp_score
            + QUALITY_WEIGHTS["intrinsics_stability"] * intr_score
        )


def view_direction(frame, object_center):
    cam = frame.cam_in_ob[:3, 3]
    direction = object_center - cam
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        direction = -frame.cam_in_ob[:3, 2]
        norm = np.linalg.norm(direction)
    return direction / max(norm, 1e-9)


def select_reference_views(frames, num_views, diag):
    clean = [fr for fr in frames if fr.quality.accepted]
    if not clean:
        return [], {"min_pairwise_angle_deg": math.nan, "median_pairwise_angle_deg": math.nan}
    points = [fr.points_ob for fr in clean if fr.points_ob is not None and len(fr.points_ob) > 0]
    object_center = np.median(np.concatenate(points, axis=0), axis=0) if points else np.median([fr.cam_in_ob[:3, 3] for fr in clean], axis=0)
    target = len(clean) if str(num_views) == "all" else min(int(num_views), len(clean))
    first_pool = sorted(clean, key=lambda fr: (fr.quality.mask_ratio, fr.quality.quality_score), reverse=True)
    selected = [first_pool[0]]
    rows = []

    while len(selected) < target:
        best = None
        best_tuple = (-1.0, -1.0)
        best_detail = None
        for fr in clean:
            if fr in selected:
                continue
            d_view = view_direction(fr, object_center)
            min_rot = math.inf
            min_view = math.inf
            for chosen in selected:
                min_rot = min(min_rot, rotation_angle(fr.cam_in_ob[:3, :3], chosen.cam_in_ob[:3, :3]))
                c_view = view_direction(chosen, object_center)
                min_view = min(min_view, math.acos(float(np.clip(np.dot(d_view, c_view), -1.0, 1.0))))
            diversity = 0.5 * (min_rot / math.pi) + 0.5 * (min_view / math.pi)
            score = diversity + 0.05 * fr.quality.quality_score
            if (score, fr.quality.quality_score) > best_tuple:
                best_tuple = (score, fr.quality.quality_score)
                best = fr
                best_detail = (min_rot, min_view, score)
        selected.append(best)
        d = view_direction(best, object_center)
        rows.append(
            {
                "selection_rank": len(selected) - 1,
                "timestamp": best.timestamp,
                "quality_score": best.quality.quality_score,
                "mask_ratio": best.quality.mask_ratio,
                "camera_position_x": best.cam_in_ob[0, 3],
                "camera_position_y": best.cam_in_ob[1, 3],
                "camera_position_z": best.cam_in_ob[2, 3],
                "view_direction_x": d[0],
                "view_direction_y": d[1],
                "view_direction_z": d[2],
                "min_rotation_distance_deg": math.degrees(best_detail[0]),
                "min_view_direction_distance_deg": math.degrees(best_detail[1]),
                "selection_score": best_detail[2],
            }
        )

    first = selected[0]
    d = view_direction(first, object_center)
    rows.insert(
        0,
        {
            "selection_rank": 0,
            "timestamp": first.timestamp,
            "quality_score": first.quality.quality_score,
            "mask_ratio": first.quality.mask_ratio,
            "camera_position_x": first.cam_in_ob[0, 3],
            "camera_position_y": first.cam_in_ob[1, 3],
            "camera_position_z": first.cam_in_ob[2, 3],
            "view_direction_x": d[0],
            "view_direction_y": d[1],
            "view_direction_z": d[2],
            "min_rotation_distance_deg": math.nan,
            "min_view_direction_distance_deg": math.nan,
            "selection_score": first.quality.quality_score,
        },
    )
    with open(diag / "view_selection.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for rank, fr in enumerate(selected):
        fr.quality.selected = True
        fr.output_index = rank

    pair_angles = []
    for i in range(len(selected)):
        vi = view_direction(selected[i], object_center)
        for j in range(i + 1, len(selected)):
            vj = view_direction(selected[j], object_center)
            pair_angles.append(math.degrees(math.acos(float(np.clip(np.dot(vi, vj), -1.0, 1.0)))))
    return selected, {
        "formula": "max-min farthest sampling with D = 0.5 * SO3_geodesic(R_i,R_j)/pi + 0.5 * view_direction_angle/pi; candidate score = min_j(D_ij) + 0.05 * quality_score",
        "object_center_source": "median XYZ of clean masked depth points transformed by Polycam corrected poses into the common observation frame",
        "min_pairwise_angle_deg": float(np.min(pair_angles)) if pair_angles else math.nan,
        "median_pairwise_angle_deg": float(np.median(pair_angles)) if pair_angles else math.nan,
    }


def validate_output_from_disk(ob, expected_count, target_shape, expected_K, final_alignment):
    errors = []
    rgb_files = sorted((ob / "rgb").glob("*.png"))
    depth_files = sorted((ob / "depth_enhanced").glob("*.png"))
    mask_files = sorted((ob / "mask").glob("*.png"))
    pose_files = sorted((ob / "cam_in_ob").glob("*.txt"))
    if not (len(rgb_files) == len(depth_files) == len(mask_files) == len(pose_files) == expected_count):
        errors.append(f"count mismatch rgb={len(rgb_files)} depth={len(depth_files)} mask={len(mask_files)} pose={len(pose_files)} expected={expected_count}")
    K = np.loadtxt(ob / "K.txt").reshape(3, 3)
    H, W = target_shape
    if not validate_K(K, W, H):
        errors.append("K validation failed")
    if not np.allclose(K, expected_K, rtol=1e-5, atol=1e-4):
        errors.append("K scaling check failed against computed median scaled K")
    for i, rgb_file in enumerate(rgb_files):
        stem = rgb_file.stem
        rgb = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        dep = cv2.imread(str(ob / "depth_enhanced" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(ob / "mask" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(ob / "cam_in_ob" / f"{stem}.txt").reshape(4, 4)
        if rgb is None or dep is None or mask is None:
            errors.append(f"cannot read frame {stem}")
            continue
        if rgb.shape[:2] != (H, W) or dep.shape != (H, W) or mask.shape[:2] != (H, W):
            errors.append(f"resolution mismatch frame {stem}: rgb={rgb.shape[:2]} depth={dep.shape} mask={mask.shape[:2]}")
        if dep.dtype != np.uint16:
            errors.append(f"depth frame {stem} dtype {dep.dtype}, expected uint16")
        if int((dep > 0).sum()) == 0:
            errors.append(f"depth frame {stem} all zero")
        mask_ratio = float((mask > 0).mean())
        if mask_ratio <= 0.0 or mask_ratio >= 0.95:
            errors.append(f"mask frame {stem} invalid ratio {mask_ratio:.4f}")
        if float(((dep > 0) & (mask > 0)).sum()) / max(1, int((mask > 0).sum())) < 0.05:
            errors.append(f"mask/depth intersection too low frame {stem}")
        if not validate_pose(pose):
            errors.append(f"pose frame {stem} invalid")
    if not final_alignment.get("pass", False):
        errors.append("final pose alignment failed")
    return errors


def write_quality_tables(frames, diag):
    rows = []
    for fr in frames:
        q = fr.quality
        row = q.__dict__.copy()
        row["reject_reasons"] = ";".join(q.reject_reasons)
        rows.append(row)
    with open(diag / "frame_quality.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(diag / "frame_quality.json", "w") as f:
        json.dump(rows, f, indent=2)


def print_report(report):
    print("========================================")
    print("Polycam -> FoundationPose Model-Free")
    print("========================================")
    print(f"Raw paired frames: {report['pairing']['fully_paired_frames']}")
    print(f"Basic validation passed: {report['counts']['basic_validation_passed']}")
    print("Rejected:")
    for key in ["blur", "exposure", "depth", "confidence", "mask", "K", "pose", "pose_alignment"]:
        print(f"{key}: {report['rejected'].get(key, 0)}")
    print(f"Clean candidate pool: {report['counts']['clean_candidate_pool']}")
    print("========================================")
    print("View Selection")
    print("========================================")
    print("Algorithm: FoundationPose-style max-min farthest-view sampling")
    print(f"Requested views: {report['view_selection']['requested_views']}")
    print(f"Selected views: {report['view_selection']['selected_views']}")
    print(f"Minimum pairwise view angle: {report['view_selection']['min_pairwise_angle_deg']:.3f} deg")
    print(f"Median pairwise view angle: {report['view_selection']['median_pairwise_angle_deg']:.3f} deg")
    print("========================================")
    print("Final Validation")
    print("========================================")
    print(f"RGB count: {report['final_validation']['rgb_count']}")
    print(f"Depth count: {report['final_validation']['depth_count']}")
    print(f"Mask count: {report['final_validation']['mask_count']}")
    print(f"Pose count: {report['final_validation']['pose_count']}")
    print(f"Resolution: {report['resolution']['target'][0]}x{report['resolution']['target'][1]} {'PASS' if report['final_validation']['resolution_pass'] else 'FAIL'}")
    print(f"Depth uint16/mm: {'PASS' if report['final_validation']['depth_uint16_mm_pass'] else 'FAIL'}")
    print(f"K scaling: {'PASS' if report['final_validation']['K_scaling_pass'] else 'FAIL'}")
    print(f"Pose matrices: {'PASS' if report['final_validation']['pose_matrices_pass'] else 'FAIL'}")
    print(f"Pose alignment: {'PASS' if report['final_validation']['pose_alignment_pass'] else 'FAIL'}")
    print(f"View diversity: {'PASS' if report['final_validation']['view_diversity_pass'] else 'FAIL'}")
    if report["ready_for_reconstruction"]:
        print("READY FOR RECONSTRUCTION")
    else:
        print("NOT READY FOR RECONSTRUCTION")


def convert(args):
    defaults = {
        "max_mask_border_contact": 0.05,
        "min_object_depth_coverage": 0.20,
        "max_alignment_points": 2500,
        "pose_alignment_iterations": 3,
        "pose_mad_factor": 3.0,
        "max_pose_alignment_error_mm": None,
        "max_intrinsics_deviation": 0.02,
        "min_sharpness_percentile": 10.0,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    raw = Path(args.input).expanduser().resolve()
    key = raw / "keyframes" if (raw / "keyframes").is_dir() else raw
    out_root = Path(args.output).expanduser().resolve()
    ob = out_root / "ob_0000001"
    diag = ob / "diagnostics"
    if ob.exists() and getattr(args, "force", False):
        shutil.rmtree(ob)
    for d in ["rgb", "depth_enhanced", "mask", "cam_in_ob", "diagnostics/mask_overlay", "diagnostics/depth_preview", "diagnostics/yolo_overlay"]:
        (ob / d).mkdir(parents=True, exist_ok=True)

    imgs, dup_img = list_by_stem(key / "corrected_images", {".jpg", ".jpeg", ".png"})
    cams, dup_cam = list_by_stem(key / "corrected_cameras", {".json"})
    depths, dup_depth = list_by_stem(key / "depth", {".png"})
    confs, dup_conf = list_by_stem(key / "confidence", {".png"})
    all_stems = set(imgs) | set(cams) | set(depths) | set(confs)
    common = sorted(set(imgs) & set(cams) & set(depths) & set(confs), key=timestamp_key)
    missing = {
        "rgb_missing": sorted(all_stems - set(imgs), key=timestamp_key),
        "camera_missing": sorted(all_stems - set(cams), key=timestamp_key),
        "depth_missing": sorted(all_stems - set(depths), key=timestamp_key),
        "confidence_missing": sorted(all_stems - set(confs), key=timestamp_key),
        "duplicate_timestamps": sorted(set(dup_img + dup_cam + dup_depth + dup_conf), key=timestamp_key),
    }
    if not common:
        raise RuntimeError("no complete Polycam frame pairs found")

    sample_rgb = cv2.imread(str(imgs[common[0]]), cv2.IMREAD_COLOR)
    sample_depth = cv2.imread(str(depths[common[0]]), cv2.IMREAD_UNCHANGED)
    if sample_rgb is None or sample_depth is None or sample_depth.dtype != np.uint16:
        raise RuntimeError("sample RGB/depth read failed or depth is not uint16 mm")
    target_h, target_w = sample_depth.shape[:2]
    src_h, src_w = sample_rgb.shape[:2]
    sx, sy = target_w / float(src_w), target_h / float(src_h)

    frames = []
    scaled_Ks = []
    raw_K_values = []
    previous_pose = None
    yolo_model = None
    yolo_cls = None
    if args.mask_mode == "yolo":
        yolo_model = load_yolo_model(args.yolo_model)
        yolo_cls = yolo_class_id(yolo_model, args.yolo_class)
        print(f"YOLO segmentation mode: model={args.yolo_model}, class={args.yolo_class} id={yolo_cls}")

    for ts in common:
        q = QualityRecord(timestamp=ts)
        try:
            camera_data, K_raw, T_poly = read_camera(cams[ts])
        except Exception as exc:
            q.accepted = False
            q.reject_reasons.append(f"invalid_camera:{exc}")
            continue
        K = K_raw.copy()
        K[0] *= sx
        K[1] *= sy
        raw_K_values.append([K_raw[0, 0], K_raw[1, 1], K_raw[0, 2], K_raw[1, 2]])
        scaled_Ks.append(K)
        cam_in_ob = T_poly @ GLCAM_IN_CVCAM
        if not validate_K(K, target_w, target_h):
            q.accepted = False
            q.reject_reasons.append("invalid_K")
        if not validate_pose(cam_in_ob):
            q.accepted = False
            q.reject_reasons.append("invalid_pose")
        if previous_pose is not None:
            q.pose_translation_delta = float(np.linalg.norm(cam_in_ob[:3, 3] - previous_pose[:3, 3]))
            q.pose_rotation_delta = math.degrees(rotation_angle(previous_pose[:3, :3], cam_in_ob[:3, :3]))
            if q.pose_translation_delta > 0.35 or q.pose_rotation_delta > 45:
                q.reject_reasons.append("possible_pose_jump")
        previous_pose = cam_in_ob

        rgb = cv2.imread(str(imgs[ts]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depths[ts]), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(confs[ts]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or conf is None:
            q.accepted = False
            q.reject_reasons.append("read_failed")
            continue
        if depth.dtype != np.uint16:
            q.accepted = False
            q.reject_reasons.append("wrong_depth_dtype")
        if depth.shape != (target_h, target_w) or conf.shape != (target_h, target_w):
            q.accepted = False
            q.reject_reasons.append("wrong_depth_confidence_resolution")
        rgb_small = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA) if rgb.shape[:2] != (target_h, target_w) else rgb.copy()
        if rgb_small.shape[:2] != depth.shape or conf.shape != depth.shape:
            q.accepted = False
            q.reject_reasons.append("resolution_mismatch")

        depth_f, high_conf, med_conf, low_conf = confidence_filter(depth, conf, args.confidence)
        q.depth_valid_ratio = float((depth_f > 0).mean())
        q.high_confidence_ratio = float(high_conf.mean())
        q.medium_confidence_ratio = float(med_conf.mean())
        q.low_confidence_ratio = float(low_conf.mean())
        q.sharpness = sharpness_score(rgb_small)
        q.mean_luminance, q.dark_pixel_ratio, q.bright_pixel_ratio, q.exposure_score, severe_exposure = exposure_stats(rgb_small)
        if severe_exposure:
            q.accepted = False
            q.reject_reasons.append("exposure")

        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        try:
            if args.mask_mode == "existing":
                src = cv2.imread(str(key / "mask" / f"{ts}.png"), cv2.IMREAD_UNCHANGED)
                if src is None:
                    raise RuntimeError("missing existing mask")
                mask = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            elif args.mask_mode == "yolo":
                raw_mask = yolo_segmentation_mask(yolo_model, yolo_cls, rgb, depth.shape, conf=args.yolo_conf, iou=args.yolo_iou, device=args.yolo_device)
                mask = raw_mask
                yolo_overlay = rgb_small.copy()
                blue = np.zeros_like(yolo_overlay)
                blue[:, :, 0] = 255
                cv2.imwrite(str(diag / "yolo_overlay" / f"{ts}.png"), np.where(mask[..., None] > 0, cv2.addWeighted(yolo_overlay, 0.55, blue, 0.45, 0), yolo_overlay))
            else:
                mask = auto_depth_mask(depth_f)
        except Exception as exc:
            q.accepted = False
            q.reject_reasons.append(f"mask_generation_failed:{exc}")
        mask = remove_small_components(mask, min_area=16)
        q.mask_ratio = float((mask > 0).mean())
        q.mask_border_contact = mask_border_contact(mask)
        safe_mask = make_depth_safe_mask(mask)
        q.mask_depth_valid_ratio = float(((depth_f > 0) & (safe_mask > 0)).sum() / max(1, int((safe_mask > 0).sum())))
        if q.mask_ratio <= 0.0:
            q.accepted = False
            q.reject_reasons.append("empty_mask")
        if q.mask_ratio < args.min_mask_ratio:
            q.accepted = False
            q.reject_reasons.append("mask_ratio_too_low")
        if q.mask_ratio > args.max_mask_ratio:
            q.accepted = False
            q.reject_reasons.append("mask_ratio_too_high")
        if q.mask_border_contact > args.max_mask_border_contact:
            q.reject_reasons.append("mask_border_contact")
        if q.mask_depth_valid_ratio < args.min_object_depth_coverage:
            q.accepted = False
            q.reject_reasons.append("low_object_depth_coverage")
        if q.depth_valid_ratio < args.min_valid_depth:
            q.accepted = False
            q.reject_reasons.append("low_depth_valid_ratio")

        pts_cam = depth_to_points(depth_f, safe_mask, K, max_points=args.max_alignment_points, stride=1)
        pts_ob = transform_points(cam_in_ob, pts_cam)
        frames.append(CandidateFrame(ts, imgs[ts], depths[ts], confs[ts], cams[ts], K_raw, K, T_poly, cam_in_ob, rgb_small, depth_f, conf, mask, safe_mask, pts_ob, q))

    if not frames:
        raise RuntimeError("no frames survived initial loading")

    K_stack = np.stack([fr.K for fr in frames])
    K_final = np.median(K_stack, axis=0)
    raw_K_array = np.array(raw_K_values, dtype=np.float64) if raw_K_values else np.empty((0, 4))
    for fr in frames:
        vals = np.array([fr.K[0, 0], fr.K[1, 1], fr.K[0, 2], fr.K[1, 2]])
        ref = np.array([K_final[0, 0], K_final[1, 1], K_final[0, 2], K_final[1, 2]])
        fr.quality.intrinsics_deviation = float(np.max(np.abs(vals - ref) / np.maximum(np.abs(ref), 1e-9)))
        if fr.quality.intrinsics_deviation > args.max_intrinsics_deviation:
            fr.quality.accepted = False
            fr.quality.reject_reasons.append("intrinsics_outlier")
        elif fr.quality.intrinsics_deviation > 0.01:
            fr.quality.reject_reasons.append("intrinsics_warning")

    sharp_values = np.array([fr.quality.sharpness for fr in frames], dtype=np.float64)
    sharp_floor = float(np.percentile(sharp_values, args.min_sharpness_percentile))
    med_sharp, mad_sharp = robust_mad(sharp_values)
    severe_floor = max(0.0, min(sharp_floor, med_sharp - 3.0 * max(mad_sharp, 0.0))) if np.isfinite(med_sharp) else sharp_floor
    for fr in frames:
        if fr.quality.sharpness < severe_floor:
            fr.quality.accepted = False
            fr.quality.reject_reasons.append("rgb_blur")
        elif fr.quality.sharpness < sharp_floor:
            fr.quality.reject_reasons.append("rgb_soft")

    alignment_all = compute_pose_alignment(
        frames,
        diag,
        "all",
        iterations=args.pose_alignment_iterations,
        pose_mad_factor=args.pose_mad_factor,
        max_pose_error_mm=args.max_pose_alignment_error_mm,
    )
    write_ply(diag / "pose_alignment_filtered.ply", [fr.points_ob for fr in frames if fr.quality.accepted])
    compute_quality_scores(frames, max(sharp_floor, 1.0))
    clean = [fr for fr in frames if fr.quality.accepted]
    if len(clean) < 3:
        write_quality_tables(frames, diag)
        raise RuntimeError(f"too few clean candidate frames ({len(clean)}) after quality filtering")

    selected, selection_stats = select_reference_views(frames, args.num_views, diag)
    if str(args.num_views).isdigit() and len(selected) < int(args.num_views):
        print(f"WARNING: Requested {args.num_views} reference views, but only {len(selected)} valid high-quality views are available.")

    np.savetxt(ob / "K.txt", K_final)
    for folder in ["rgb", "depth_enhanced", "mask", "cam_in_ob"]:
        for p in (ob / folder).glob("*"):
            if p.is_file():
                p.unlink()
    for out_idx, fr in enumerate(selected):
        name = f"{out_idx:06d}"
        cv2.imwrite(str(ob / "rgb" / f"{name}.png"), fr.rgb_small)
        cv2.imwrite(str(ob / "depth_enhanced" / f"{name}.png"), fr.depth_mm.astype(np.uint16))
        cv2.imwrite(str(ob / "mask" / f"{name}.png"), fr.mask_original.astype(np.uint8))
        np.savetxt(ob / "cam_in_ob" / f"{name}.txt", fr.cam_in_ob)
        overlay = fr.rgb_small.copy()
        red = np.zeros_like(overlay)
        red[:, :, 2] = 255
        cv2.imwrite(str(diag / "mask_overlay" / f"{name}_{fr.timestamp}.png"), np.where(fr.mask_original[..., None] > 0, cv2.addWeighted(overlay, 0.55, red, 0.45, 0), overlay))
        cv2.imwrite(str(diag / "depth_preview" / f"{name}_{fr.timestamp}.png"), cv2.convertScaleAbs(fr.depth_mm, alpha=255.0 / max(1, int(fr.depth_mm.max()))))

    (ob / "select_frames.yml").write_text("frames:\n" + "".join(f"  - {i}\n" for i in range(len(selected))))
    make_contactsheet(frames, diag / "all_candidates.jpg", lambda fr: f"{fr.timestamp}\nq={fr.quality.quality_score:.3f}\nmask={fr.quality.mask_ratio:.3f}")
    for reason, filename in [("rgb_blur", "rejected_blur.jpg"), ("low_object_depth_coverage", "rejected_depth.jpg"), ("empty_mask", "rejected_mask.jpg"), ("pose_alignment_outlier", "rejected_pose.jpg")]:
        subset = [fr for fr in frames if any(reason in r for r in fr.quality.reject_reasons)]
        make_contactsheet(subset, diag / filename, lambda fr: f"{fr.timestamp}\n{reason}\nq={fr.quality.quality_score:.3f}", max_items=60)
    make_contactsheet(
        selected,
        diag / f"selected_views_{len(selected)}.jpg",
        lambda fr: f"#{fr.output_index + 1:02d} {fr.timestamp}\nq={fr.quality.quality_score:.3f} mask={fr.quality.mask_ratio:.3f}\ndepth={fr.quality.mask_depth_valid_ratio:.3f}\npose={fr.quality.pose_alignment_error * 1000.0:.1f}mm",
    )

    final_alignment = compute_pose_alignment(selected, diag, "final_24", iterations=1, pose_mad_factor=args.pose_mad_factor, max_pose_error_mm=args.max_pose_alignment_error_mm)
    write_ply(diag / "final_24_pose_alignment.ply", [fr.points_ob for fr in selected])
    final_errors = validate_output_from_disk(ob, len(selected), (target_h, target_w), K_final, final_alignment)

    write_quality_tables(frames, diag)
    reject_buckets = {
        "blur": set(),
        "exposure": set(),
        "depth": set(),
        "confidence": set(),
        "mask": set(),
        "K": set(),
        "pose": set(),
        "pose_alignment": set(),
    }
    for fr in frames:
        if fr.quality.accepted:
            continue
        for reason in fr.quality.reject_reasons:
            if reason.startswith("intrinsics"):
                key = "K"
            elif "pose_alignment" in reason:
                key = "pose_alignment"
            elif "pose" in reason:
                key = "pose"
            elif "depth" in reason:
                key = "depth"
            elif "confidence" in reason:
                key = "confidence"
            elif "mask" in reason:
                key = "mask"
            elif "blur" in reason or "soft" in reason:
                key = "blur"
            elif "exposure" in reason:
                key = "exposure"
            else:
                key = reason
            reject_buckets.setdefault(key, set()).add(fr.timestamp)
    reject_counts = {key: len(value) for key, value in reject_buckets.items()}

    report = {
        "input": str(raw),
        "output": str(ob),
        "pairing": {
            "RGB_count": len(imgs),
            "Camera_count": len(cams),
            "Depth_count": len(depths),
            "Confidence_count": len(confs),
            "fully_paired_frames": len(common),
            "missing_frames": missing,
        },
        "resolution": {"source_rgb": [src_w, src_h], "target": [target_w, target_h], "scale": [sx, sy]},
        "K_raw_stats": {
            "median": np.median(raw_K_array, axis=0).tolist() if raw_K_array.size else None,
            "mad": np.median(np.abs(raw_K_array - np.median(raw_K_array, axis=0)), axis=0).tolist() if raw_K_array.size else None,
            "min": raw_K_array.min(axis=0).tolist() if raw_K_array.size else None,
            "max": raw_K_array.max(axis=0).tolist() if raw_K_array.size else None,
        },
        "K_scaled": K_final.tolist(),
        "depth_unit": "uint16 millimetres; invalid depth is 0; confidence filtering zeroes rejected depth pixels without resizing depth",
        "pose_convention": "Polycam corrected t_ij is treated as OpenGL/ARKit camera-to-world; cam_in_ob = T_polycam @ glcam_in_cvcam, matching BundleSDF run_neural_object_field which converts cam_in_obs back to OpenGL with cam_in_obs @ glcam_in_cvcam.",
        "quality_weights": QUALITY_WEIGHTS,
        "counts": {
            "loaded_frames": len(frames),
            "basic_validation_passed": len([fr for fr in frames if not any(r in fr.quality.reject_reasons for r in ["invalid_K", "invalid_pose", "wrong_depth_dtype", "resolution_mismatch"])]),
            "clean_candidate_pool": len(clean),
        },
        "rejected": reject_counts,
        "pose_alignment": alignment_all,
        "view_selection": {
            "requested_views": args.num_views,
            "selected_views": len(selected),
            "selected_timestamps": [fr.timestamp for fr in selected],
            **selection_stats,
        },
        "final_validation": {
            "rgb_count": len(list((ob / "rgb").glob("*.png"))),
            "depth_count": len(list((ob / "depth_enhanced").glob("*.png"))),
            "mask_count": len(list((ob / "mask").glob("*.png"))),
            "pose_count": len(list((ob / "cam_in_ob").glob("*.txt"))),
            "resolution_pass": not any("resolution" in e for e in final_errors),
            "depth_uint16_mm_pass": not any("depth" in e for e in final_errors),
            "K_scaling_pass": not any("K" in e for e in final_errors),
            "pose_matrices_pass": not any("pose frame" in e for e in final_errors),
            "pose_alignment_pass": final_alignment.get("pass", False),
            "view_diversity_pass": len(selected) >= min(3, int(args.num_views) if str(args.num_views).isdigit() else len(selected)),
            "errors": final_errors,
            "final_pose_alignment": final_alignment,
        },
        "ready_for_reconstruction": len(final_errors) == 0 and final_alignment.get("pass", False),
        "frames": [
            {
                "timestamp": fr.timestamp,
                "source": {
                    "rgb": str(fr.rgb_path),
                    "depth": str(fr.depth_path),
                    "confidence": str(fr.confidence_path),
                    "camera": str(fr.camera_path),
                },
                "output_index": fr.output_index,
                "K": fr.K.tolist(),
                "cam_in_ob": fr.cam_in_ob.tolist(),
                "quality": {**fr.quality.__dict__, "reject_reasons": fr.quality.reject_reasons},
            }
            for fr in frames
        ],
    }
    with open(ob / "manifest.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(diag / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(diag / "selected_views.txt", "w") as f:
        for fr in selected:
            f.write(f"{fr.output_index:06d} {fr.timestamp} quality={fr.quality.quality_score:.6f} mask={fr.quality.mask_ratio:.6f} pose_error_m={fr.quality.pose_alignment_error}\n")
    with open(diag / "pose_convention.txt", "w") as f:
        f.write(report["pose_convention"] + "\n")
        f.write("glcam_in_cvcam = diag(1,-1,-1,1)\n")
    print_report(report)
    if not report["ready_for_reconstruction"]:
        raise RuntimeError("Final validation gate failed; not running Neural Object Field. See diagnostics/validation_report.json")
    return ob
