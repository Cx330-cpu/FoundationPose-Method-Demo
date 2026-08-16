#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


GLCAM_IN_CVCAM = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)


@dataclass
class Frame:
    timestamp: str
    rgb_path: Path
    depth_path: Path
    confidence_path: Path
    camera_path: Path
    K_raw: np.ndarray
    K: np.ndarray
    polycam_cam_in_world: np.ndarray
    cam_in_ob: np.ndarray
    blur_score: float
    weakly_connected: bool
    quality: dict
    selected: bool = False
    reject_reason: str = ""
    mask_error: str = ""


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


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


def list_by_stem(directory, suffixes=None):
    out = {}
    dup = []
    if not directory.is_dir():
        return out, dup
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if suffixes and p.suffix.lower() not in suffixes:
            continue
        if p.stem in out:
            dup.append(p.stem)
        out[p.stem] = p
    return out, sorted(set(dup))


def depth_to_points(depth_m, K, stride=2):
    ys, xs = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    z = depth_m[ys, xs]
    valid = z > 0.001
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    pix = np.stack([xs, ys], axis=1)
    return pts, pix


def transform_points(T, pts):
    if len(pts) == 0:
        return pts
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
    return (T @ pts_h.T).T[:, :3]


def project_points(world_pts, world_from_cv, K, shape):
    H, W = shape[:2]
    cv_from_world = np.linalg.inv(world_from_cv)
    pts_cam = transform_points(cv_from_world, world_pts)
    z = pts_cam[:, 2]
    valid = z > 0.001
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    u = np.round(K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]).astype(np.int32)
    v = np.round(K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]).astype(np.int32)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[v[valid], u[valid]] = 255
    return mask


def confidence_filter(depth, conf, mode):
    if mode == "all":
        keep = np.ones(conf.shape, dtype=bool)
    elif mode == "high":
        keep = conf == conf.max()
    else:
        keep = conf > 0
    out = depth.copy()
    out[~keep] = 0
    return out, keep


def clean_mask(mask, min_area=64):
    mask = (mask > 0).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    if areas.max() < min_area:
        return np.zeros_like(mask)
    return (labels == keep).astype(np.uint8) * 255


def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "YOLO mask mode requires the ultralytics package inside this environment. "
            "Install it in the 5070ti Docker image, then rerun with --mask_mode yolo."
        ) from exc
    try:
        return YOLO(model_path)
    except Exception as exc:
        # PyTorch 2.6+ defaults torch.load(weights_only=True). Older
        # ultralytics checkpoints need their model classes allowlisted.
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
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        for idx, name in names.items():
            if str(name).lower() == class_name.lower():
                return int(idx)
    elif isinstance(names, (list, tuple)):
        for idx, name in enumerate(names):
            if str(name).lower() == class_name.lower():
                return idx
    raise RuntimeError(f"YOLO class '{class_name}' was not found in model names: {names}")


def yolo_segmentation_mask(model, class_id, rgb_bgr, target_shape, conf=0.25, iou=0.7, device=None):
    results = model.predict(rgb_bgr[..., ::-1], conf=conf, iou=iou, device=device, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no result")
    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise RuntimeError("YOLO result has no segmentation masks")
    if result.boxes is None or result.boxes.cls is None:
        raise RuntimeError("YOLO result has masks but no class boxes")

    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(cls))
    masks = result.masks.data.detach().cpu().numpy()
    candidates = []
    H, W = target_shape[:2]
    for i, cid in enumerate(cls):
        if cid != class_id:
            continue
        mask = masks[i]
        mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        center = np.array([xs.mean() / W, ys.mean() / H])
        center_score = 1.0 - min(1.0, np.linalg.norm(center - np.array([0.5, 0.5])) / 0.7)
        area = mask.mean()
        score = float(confs[i]) + 0.25 * center_score + 0.15 * min(area / 0.2, 1.0)
        candidates.append((score, mask))
    if not candidates:
        raise RuntimeError(f"YOLO found segmentation masks, but none for class id {class_id}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1].astype(np.uint8) * 255


def sharpness(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_contactsheet(images, labels, out_file, tile_w=192):
    if not images:
        return
    thumbs = []
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        scale = tile_w / w
        tile = cv2.resize(img, (tile_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(tile)
    cols = min(6, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    th, tw = thumbs[0].shape[:2]
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th : r * th + th, c * tw : c * tw + tw] = tile
    cv2.imwrite(str(out_file), sheet)


def build_object_cloud(frames, max_frames=32):
    if o3d is None:
        raise RuntimeError("open3d is required for auto mask generation")
    samples = frames if len(frames) <= max_frames else frames[:: max(1, len(frames) // max_frames)][:max_frames]
    pts_all = []
    colors_all = []
    for fr in samples:
        rgb = cv2.imread(str(fr.rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(fr.depth_path), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(fr.confidence_path), cv2.IMREAD_UNCHANGED)
        depth, _ = confidence_filter(depth, conf, "medium-high")
        pts, pix = depth_to_points(depth.astype(np.float64) / 1000.0, fr.K, stride=2)
        if len(pts) == 0:
            continue
        pts_w = transform_points(fr.cam_in_ob, pts)
        c = rgb[pix[:, 1].astype(int), pix[:, 0].astype(int), ::-1] / 255.0
        pts_all.append(pts_w)
        colors_all.append(c)
    if not pts_all:
        raise RuntimeError("no valid depth points for auto mask")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors_all))
    pcd = pcd.voxel_down_sample(0.006)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(pcd.points) < 200:
        raise RuntimeError("auto mask failed: scene cloud has too few points")

    pcd_np = pcd
    # Polycam captures often contain more than one support/background plane
    # near the object. Remove several dominant planes before clustering, but
    # stop before eating compact object geometry.
    try:
        for _ in range(4):
            if len(pcd_np.points) < 500:
                break
            _, inliers = pcd_np.segment_plane(distance_threshold=0.012, ransac_n=3, num_iterations=1000)
            ratio = len(inliers) / max(1, len(pcd_np.points))
            if ratio < 0.08 or ratio > 0.85:
                break
            pcd_np = pcd_np.select_by_index(inliers, invert=True)
    except Exception:
        pass

    labels = np.asarray(pcd_np.cluster_dbscan(eps=0.035, min_points=25, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        raise RuntimeError("auto mask failed: no object cluster after plane removal")
    pts = np.asarray(pcd_np.points)
    cams = np.array([f.cam_in_ob[:3, 3] for f in frames])
    traj_center = np.median(cams, axis=0)
    best_label = None
    best_score = -1e18
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        ids = labels == label
        cluster = pts[ids]
        extent = cluster.max(axis=0) - cluster.min(axis=0)
        diag = float(np.linalg.norm(extent))
        count = int(ids.sum())
        dist = float(np.linalg.norm(cluster.mean(axis=0) - traj_center))
        if count < 80 or diag < 0.03 or diag > 0.8:
            continue
        visibility = 0
        center_hits = []
        area_scores = []
        sample_frames = frames[:: max(1, len(frames) // 20)][:20]
        for fr in sample_frames:
            proj = project_points(cluster, fr.cam_in_ob, fr.K, (192, 256))
            ys, xs = np.where(proj > 0)
            if len(xs) == 0:
                continue
            visibility += 1
            cx, cy = xs.mean() / 256.0, ys.mean() / 192.0
            center_hits.append(max(0.0, 1.0 - np.linalg.norm(np.array([cx, cy]) - np.array([0.5, 0.5])) / 0.55))
            area_scores.append(min(len(xs) / float(256 * 192), 0.08) / 0.08)
        if visibility < max(6, int(0.3 * len(sample_frames))):
            continue
        compact = min(count / max(diag, 1e-6), 2500.0)
        score = compact + 1800.0 * np.mean(center_hits) + 700.0 * np.mean(area_scores) + 120.0 * visibility - 80.0 * dist
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None:
        raise RuntimeError("auto mask failed: no compact cluster with reasonable object size")
    obj = pcd_np.select_by_index(np.where(labels == best_label)[0])
    obj = obj.voxel_down_sample(0.003)
    if len(obj.points) < 80:
        raise RuntimeError("auto mask failed: object cluster too small")
    return obj, pcd


def select_views(frames, num_views):
    good = [f for f in frames if not f.reject_reason]
    if num_views == "all":
        for f in good:
            f.selected = True
        return good
    n = min(int(num_views), len(good))
    if n <= 0:
        return []
    scores = np.array([f.quality["mask_ratio"] * 4 + f.quality["valid_depth_ratio"] + min(f.quality["sharpness"], 500.0) / 500.0 for f in good])
    selected = [good[int(np.argmax(scores))]]
    while len(selected) < n:
        best = None
        best_score = -1e18
        for fr in good:
            if fr in selected:
                continue
            view = -fr.polycam_cam_in_world[:3, 2]
            view = view / max(np.linalg.norm(view), 1e-9)
            dists = []
            for s in selected:
                sv = -s.polycam_cam_in_world[:3, 2]
                sv = sv / max(np.linalg.norm(sv), 1e-9)
                dists.append(1.0 - float(np.dot(view, sv)))
            score = min(dists) + 0.15 * scores[good.index(fr)]
            if score > best_score:
                best_score = score
                best = fr
        selected.append(best)
    for f in selected:
        f.selected = True
    return selected


def validate_ob(base_dir):
    base = Path(base_dir)
    rgb = sorted((base / "rgb").glob("*.png"))
    depth = sorted((base / "depth_enhanced").glob("*.png"))
    mask = sorted((base / "mask").glob("*.png"))
    pose = sorted((base / "cam_in_ob").glob("*.txt"))
    errors = []
    if not (len(rgb) == len(depth) == len(mask) == len(pose) and len(rgb) > 0):
        errors.append(f"count mismatch rgb={len(rgb)} depth={len(depth)} mask={len(mask)} pose={len(pose)}")
    K = np.loadtxt(base / "K.txt")
    first_shape = None
    for i, p in enumerate(rgb):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        dep = cv2.imread(str(depth[i]), cv2.IMREAD_UNCHANGED)
        ma = cv2.imread(str(mask[i]), cv2.IMREAD_UNCHANGED)
        if im is None or dep is None or ma is None:
            errors.append(f"cannot read frame {i}")
            continue
        shape = im.shape[:2]
        first_shape = first_shape or shape
        if shape != first_shape or dep.shape[:2] != shape or ma.shape[:2] != shape:
            errors.append(f"shape mismatch frame {i}: rgb={shape} depth={dep.shape[:2]} mask={ma.shape[:2]}")
        if dep.dtype != np.uint16:
            errors.append(f"depth frame {i} is {dep.dtype}, expected uint16 mm")
        if int((dep > 0).sum()) == 0:
            errors.append(f"depth frame {i} has no valid pixels")
        ratio = float((ma > 0).mean())
        if ratio < 0.002 or ratio > 0.85:
            errors.append(f"mask frame {i} ratio {ratio:.4f} outside reasonable range")
        T = np.loadtxt(pose[i]).reshape(4, 4)
        R = T[:3, :3]
        if not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5):
            errors.append(f"pose frame {i} invalid homogeneous row")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or abs(np.linalg.det(R) - 1) > 1e-3:
            errors.append(f"pose frame {i} rotation not SO(3), det={np.linalg.det(R):.6f}")
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        errors.append("K invalid")
    if first_shape:
        H, W = first_shape
        if not (0 <= K[0, 2] < W and 0 <= K[1, 2] < H):
            errors.append(f"K principal point outside image: {K[0,2]}, {K[1,2]} for {W}x{H}")
    return errors


def convert(args):
    from tools.polycam_quality_pipeline import convert as enhanced_convert

    return enhanced_convert(args)

    raw = Path(args.input).expanduser().resolve()
    key = raw / "keyframes" if (raw / "keyframes").is_dir() else raw
    out_root = Path(args.output).expanduser().resolve()
    ob = out_root / "ob_0000001"
    diag = ob / "diagnostics"
    if ob.exists() and args.force:
        shutil.rmtree(ob)
    ob.mkdir(parents=True, exist_ok=True)
    for d in ["rgb", "depth_enhanced", "mask", "cam_in_ob", "diagnostics/mask_overlay", "diagnostics/yolo_overlay", "diagnostics/depth_preview"]:
        (ob / d).mkdir(parents=True, exist_ok=True)

    imgs, dup_img = list_by_stem(key / "corrected_images", {".jpg", ".jpeg", ".png"})
    cams, dup_cam = list_by_stem(key / "corrected_cameras", {".json"})
    depths, dup_depth = list_by_stem(key / "depth", {".png"})
    confs, dup_conf = list_by_stem(key / "confidence", {".png"})
    common = sorted(set(imgs) & set(cams) & set(depths) & set(confs), key=lambda x: int(x) if x.isdigit() else x)
    print(f"total corrected RGB: {len(imgs)}")
    print(f"total corrected camera: {len(cams)}")
    print(f"total depth: {len(depths)}")
    print(f"total confidence: {len(confs)}")
    print(f"valid paired frames: {len(common)}")
    missing = {
        "rgb_missing": sorted((set(cams) | set(depths) | set(confs)) - set(imgs)),
        "camera_missing": sorted((set(imgs) | set(depths) | set(confs)) - set(cams)),
        "depth_missing": sorted((set(imgs) | set(cams) | set(confs)) - set(depths)),
        "confidence_missing": sorted((set(imgs) | set(cams) | set(depths)) - set(confs)),
        "duplicate_timestamps": sorted(set(dup_img + dup_cam + dup_depth + dup_conf)),
    }
    if not common:
        raise RuntimeError("no complete Polycam frame pairs found")

    raw_Ks = []
    camera_records = {}
    sizes = []
    for ts in common:
        data, K_raw, T = read_camera(cams[ts])
        camera_records[ts] = (data, K_raw, T)
        raw_Ks.append(K_raw)
        sizes.append((int(data["width"]), int(data["height"])))
    raw_Ks = np.stack(raw_Ks)
    k_min, k_max, k_med = raw_Ks.min(axis=0), raw_Ks.max(axis=0), np.median(raw_Ks, axis=0)
    rel = np.max(np.abs(k_max[:2] - k_min[:2]) / np.maximum(np.abs(k_med[:2]), 1e-9))
    if rel > 0.01:
        raise RuntimeError(f"intrinsics vary by >1% across frames: relative range={rel:.4f}")

    sample_rgb = cv2.imread(str(imgs[common[0]]), cv2.IMREAD_COLOR)
    sample_depth = cv2.imread(str(depths[common[0]]), cv2.IMREAD_UNCHANGED)
    if sample_rgb is None or sample_depth is None:
        raise RuntimeError("cannot read sample RGB/depth")
    target_h, target_w = sample_depth.shape[:2]
    src_h, src_w = sample_rgb.shape[:2]
    sx, sy = target_w / float(src_w), target_h / float(src_h)
    K_scaled = k_med.copy()
    K_scaled[0] *= sx
    K_scaled[1] *= sy
    np.savetxt(ob / "K.txt", K_scaled)

    frames = []
    manifest_frames = []
    for ts in common:
        data, K_raw, T_poly = camera_records[ts]
        rgb = cv2.imread(str(imgs[ts]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depths[ts]), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(confs[ts]), cv2.IMREAD_UNCHANGED)
        if rgb.shape[:2] != (target_h, target_w):
            rgb_small = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb
        depth_f, keep_conf = confidence_filter(depth, conf, args.confidence)
        unique_conf = np.unique(conf)
        nonzero = unique_conf[unique_conf > 0]
        med_mask = (conf > 0) & (conf < conf.max()) if len(nonzero) else np.zeros_like(conf, dtype=bool)
        quality = {
            "valid_depth_ratio": float((depth_f > 0).mean()),
            "high_confidence_ratio": float((conf == conf.max()).mean()),
            "medium_confidence_ratio": float(med_mask.mean()),
            "zero_confidence_ratio": float((conf == 0).mean()),
            "sharpness": sharpness(rgb_small),
            "mask_ratio": 0.0,
        }
        cam_in_ob = T_poly @ GLCAM_IN_CVCAM
        frames.append(
            Frame(ts, imgs[ts], depths[ts], confs[ts], cams[ts], K_raw, K_scaled, T_poly, cam_in_ob, float(data.get("blur_score", 0.0)), bool(data.get("weakly_connected", False)), quality)
        )

    yolo_model = None
    yolo_cls = None
    if args.mask_mode == "yolo":
        yolo_model = load_yolo_model(args.yolo_model)
        yolo_cls = yolo_class_id(yolo_model, args.yolo_class)
        print(f"YOLO segmentation mode: model={args.yolo_model}, class={args.yolo_class} id={yolo_cls}")

    if args.mask_mode == "auto":
        obj_cloud, scene_cloud = build_object_cloud(frames)
        o3d.io.write_point_cloud(str(diag / "object_cloud.ply"), obj_cloud)
        o3d.io.write_point_cloud(str(diag / "scene_cloud.ply"), scene_cloud)
        obj_pts = np.asarray(obj_cloud.points)
    elif args.mask_mode == "existing":
        existing = key / "mask"
        if not existing.is_dir():
            raise RuntimeError("--mask_mode existing requested, but keyframes/mask does not exist")
        obj_pts = None
    else:
        obj_pts = None

    for idx, fr in enumerate(frames):
        rgb = cv2.imread(str(fr.rgb_path), cv2.IMREAD_COLOR)
        rgb_small = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        depth = cv2.imread(str(fr.depth_path), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(fr.confidence_path), cv2.IMREAD_UNCHANGED)
        depth_f, _ = confidence_filter(depth, conf, args.confidence)
        if args.mask_mode == "auto":
            mask = project_points(obj_pts, fr.cam_in_ob, fr.K, depth_f.shape)
            mask = mask & (depth_f > 0).astype(np.uint8) * 255
            mask = clean_mask(mask)
        elif args.mask_mode == "existing":
            src = cv2.imread(str((key / "mask" / f"{fr.timestamp}.png")), cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            try:
                raw_yolo_mask = yolo_segmentation_mask(
                    yolo_model,
                    yolo_cls,
                    rgb,
                    depth_f.shape,
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    device=args.yolo_device,
                )
                mask = raw_yolo_mask & (depth_f > 0).astype(np.uint8) * 255
                mask = clean_mask(mask, min_area=32)
                yolo_overlay = rgb_small.copy()
                blue = np.zeros_like(yolo_overlay)
                blue[:, :, 0] = 255
                yolo_overlay = np.where(raw_yolo_mask[..., None] > 0, cv2.addWeighted(yolo_overlay, 0.55, blue, 0.45, 0), yolo_overlay)
                cv2.imwrite(str(diag / "yolo_overlay" / f"{fr.timestamp}.png"), yolo_overlay)
            except Exception as exc:
                fr.mask_error = str(exc)
                fr.reject_reason = "yolo_no_segmentation_mask"
                print(f"[YOLO-MASK-SKIP] {fr.timestamp}: {exc}")
                mask = np.zeros(depth_f.shape, dtype=np.uint8)
        fr.quality["mask_ratio"] = float((mask > 0).mean())
        if fr.reject_reason:
            pass
        elif fr.quality["valid_depth_ratio"] < args.min_valid_depth:
            fr.reject_reason = "depth_valid_ratio_too_low"
        elif fr.quality["mask_ratio"] < args.min_mask_ratio:
            fr.reject_reason = "mask_ratio_too_low"
        elif fr.quality["mask_ratio"] > args.max_mask_ratio:
            fr.reject_reason = "mask_ratio_too_high"
        elif fr.quality["sharpness"] < args.min_sharpness:
            fr.reject_reason = "blur_too_high"
        fr._rgb_small = rgb_small
        fr._depth_f = depth_f
        fr._mask = mask

    selected = select_views(frames, args.num_views)
    if len(selected) < 3:
        raise RuntimeError(f"too few selected frames ({len(selected)}) after quality filtering")

    overlay_imgs = []
    overlay_labels = []
    for out_idx, fr in enumerate(selected):
        name = f"{out_idx:06d}"
        cv2.imwrite(str(ob / "rgb" / f"{name}.png"), fr._rgb_small)
        cv2.imwrite(str(ob / "depth_enhanced" / f"{name}.png"), fr._depth_f.astype(np.uint16))
        cv2.imwrite(str(ob / "mask" / f"{name}.png"), fr._mask.astype(np.uint8))
        np.savetxt(ob / "cam_in_ob" / f"{name}.txt", fr.cam_in_ob)
        overlay = fr._rgb_small.copy()
        red = np.zeros_like(overlay)
        red[:, :, 2] = 255
        overlay = np.where(fr._mask[..., None] > 0, cv2.addWeighted(overlay, 0.55, red, 0.45, 0), overlay)
        cv2.imwrite(str(diag / "mask_overlay" / f"{name}_{fr.timestamp}.png"), overlay)
        cv2.imwrite(str(diag / "depth_preview" / f"{name}_{fr.timestamp}.png"), cv2.convertScaleAbs(fr._depth_f, alpha=255.0 / max(1, int(fr._depth_f.max()))))
        overlay_imgs.append(overlay)
        overlay_labels.append(f"{name} {fr.timestamp}")

    (ob / "select_frames.yml").write_text("{}\n")
    make_contactsheet(overlay_imgs, overlay_labels, diag / "selected_views_contactsheet.png")

    if o3d is not None:
        align = o3d.geometry.PointCloud()
        pts_all = []
        col_all = []
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
        sample = selected[:: max(1, len(selected) // min(12, len(selected)))]
        for i, fr in enumerate(sample[:12]):
            pts, _ = depth_to_points((fr._depth_f * (fr._mask > 0)).astype(np.float64) / 1000.0, fr.K, stride=2)
            pts_w = transform_points(fr.cam_in_ob, pts)
            pts_all.append(pts_w)
            col_all.append(np.tile(colors[i % len(colors)], (len(pts_w), 1)))
        if pts_all:
            align.points = o3d.utility.Vector3dVector(np.concatenate(pts_all))
            align.colors = o3d.utility.Vector3dVector(np.concatenate(col_all))
            o3d.io.write_point_cloud(str(diag / "pose_alignment.ply"), align.voxel_down_sample(0.004))
        traj = o3d.geometry.PointCloud()
        traj.points = o3d.utility.Vector3dVector(np.array([f.cam_in_ob[:3, 3] for f in frames]))
        o3d.io.write_point_cloud(str(diag / "camera_trajectory.ply"), traj)

    errors = validate_ob(ob)
    report = {
        "input": str(raw),
        "output": str(ob),
        "pairing": {
            "total_corrected_rgb": len(imgs),
            "total_corrected_camera": len(cams),
            "total_depth": len(depths),
            "total_confidence": len(confs),
            "valid_paired_frames": len(common),
            "missing_files": missing,
        },
        "resolution": {"source_rgb": [src_w, src_h], "target": [target_w, target_h], "scale": [sx, sy]},
        "K_raw_min": k_min.tolist(),
        "K_raw_max": k_max.tolist(),
        "K_raw_median": k_med.tolist(),
        "K_scaled": K_scaled.tolist(),
        "pose_convention": "Polycam corrected t_ij treated as OpenGL/ARKit camera-to-world. FoundationPose cam_in_ob is OpenCV camera-to-ob/world. Applied cam_in_ob = T_polycam @ glcam_in_cvcam.",
        "mask_mode": args.mask_mode,
        "yolo": {
            "model": getattr(args, "yolo_model", None),
            "class": getattr(args, "yolo_class", None),
            "confidence": getattr(args, "yolo_conf", None),
            "iou": getattr(args, "yolo_iou", None),
            "device": getattr(args, "yolo_device", None),
        } if args.mask_mode == "yolo" else None,
        "frames": [],
        "validation_errors": errors,
    }
    with open(diag / "selected_views.txt", "w") as f:
        for i, fr in enumerate(selected):
            f.write(f"{i:06d} {fr.timestamp} pos={fr.cam_in_ob[:3,3].tolist()} quality={fr.quality}\n")
    with open(diag / "pose_convention.txt", "w") as f:
        f.write(report["pose_convention"] + "\n")
        f.write("glcam_in_cvcam = diag(1,-1,-1,1)\n")
        f.write("run_neural_object_field internally computes glcam_in_obs = cam_in_obs @ glcam_in_cvcam\n")
    for i, fr in enumerate(frames):
        report["frames"].append(
            {
                "timestamp": fr.timestamp,
                "output_index": selected.index(fr) if fr in selected else None,
                "rgb_path": str(fr.rgb_path),
                "depth_path": str(fr.depth_path),
                "camera_path": str(fr.camera_path),
                "confidence_path": str(fr.confidence_path),
                "K": fr.K.tolist(),
                "camera_pose_polycam": fr.polycam_cam_in_world.tolist(),
                "cam_in_ob": fr.cam_in_ob.tolist(),
                "quality": fr.quality,
                "selected": fr.selected,
                "rejected": not fr.selected,
                "reject_reason": fr.reject_reason,
                "mask_error": fr.mask_error,
            }
        )
    with open(ob / "manifest.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(diag / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if errors:
        raise RuntimeError("FoundationPose data validation failed:\n" + "\n".join(errors[:20]))
    print(f"CONVERSION SUCCESS: {ob}")
    print(f"selected frames: {len(selected)} / {len(frames)}")
    return ob


def main():
    parser = argparse.ArgumentParser(description="Convert Polycam LiDAR raw keyframes to FoundationPose model-free reference data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_views", default="24")
    parser.add_argument("--confidence", choices=["medium-high", "high", "all"], default="medium-high")
    parser.add_argument("--mask_mode", choices=["auto", "existing", "yolo"], default="auto")
    parser.add_argument("--yolo_model", default="yolov8x-seg.pt")
    parser.add_argument("--yolo_class", default="cup")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.7)
    parser.add_argument("--yolo_device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min_valid_depth", type=float, default=0.05)
    parser.add_argument("--min_mask_ratio", type=float, default=0.003)
    parser.add_argument("--max_mask_ratio", type=float, default=0.65)
    parser.add_argument("--max_mask_border_contact", type=float, default=0.05)
    parser.add_argument("--min_object_depth_coverage", type=float, default=0.20)
    parser.add_argument("--min_sharpness", type=float, default=5.0)
    parser.add_argument("--min-sharpness-percentile", dest="min_sharpness_percentile", type=float, default=10.0)
    parser.add_argument("--max-intrinsics-deviation", dest="max_intrinsics_deviation", type=float, default=0.02)
    parser.add_argument("--pose-mad-factor", dest="pose_mad_factor", type=float, default=3.0)
    parser.add_argument("--pose-alignment-iterations", dest="pose_alignment_iterations", type=int, default=3)
    parser.add_argument("--max-pose-alignment-error-mm", dest="max_pose_alignment_error_mm", type=float, default=None)
    parser.add_argument("--max-alignment-points", dest="max_alignment_points", type=int, default=2500)
    parser.add_argument("--prepare-only", dest="prepare_only", action="store_true")
    parser.add_argument("--quality-check-only", dest="quality_check_only", action="store_true")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
