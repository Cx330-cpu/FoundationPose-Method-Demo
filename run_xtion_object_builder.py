#!/usr/bin/env python3
"""ASUS Xtion Pro Live model-free object builder for FoundationPose.

This script only prepares a metric object model/reference asset. The Xtion is
not used by the later iPhone live tracking pipeline.
"""

from __future__ import annotations

import argparse
import copy
import ctypes.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent
BUNDLESDF_DIR = REPO_ROOT / "bundlesdf"
MYCUDA_DIR = BUNDLESDF_DIR / "mycuda"
GLCAM_IN_CVCAM = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class PreflightResult:
    usb_devices: list[str] = field(default_factory=list)
    xtion_usb_visible: bool = False
    dev_bus_usb_exists: bool = False
    video_devices: list[str] = field(default_factory=list)
    openni_library: str | None = None
    primesense_available: bool = False
    opencv_available: bool = False
    open3d_available: bool = False
    trimesh_available: bool = False
    torch_available: bool = False
    torch_cuda_available: bool = False
    torch_version: str | None = None
    bundle_cuda_common: bool = False
    bundle_cuda_gridencoder: bool = False
    bundle_cuda_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class CameraInfo:
    name: str = "ASUS Xtion Pro Live"
    vendor: str | None = None
    uri: str | None = None
    rgb_resolution: tuple[int, int] | None = None
    depth_resolution: tuple[int, int] | None = None
    fps: int | None = None
    depth_registered_to_rgb: bool = False
    K: np.ndarray | None = None
    K_source: str = "unknown"


@dataclass
class CandidateFrame:
    rgb: np.ndarray
    depth_mm: np.ndarray
    mask: np.ndarray
    cam_in_ob: np.ndarray
    timestamp: float
    quality_score: float
    view_note: str = ""


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def add_bundlesdf_paths() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(BUNDLESDF_DIR))
    sys.path.insert(0, str(MYCUDA_DIR))
    for build_dir in sorted((MYCUDA_DIR / "build").glob("lib.*")):
        sys.path.insert(0, str(build_dir))


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 127, "", repr(exc)


def preflight(check_bundle_cuda: bool = True) -> PreflightResult:
    result = PreflightResult()
    result.dev_bus_usb_exists = Path("/dev/bus/usb").exists()
    result.video_devices = sorted(str(p) for p in Path("/dev").glob("video*"))

    if shutil.which("lsusb"):
        code, out, err = run_cmd(["lsusb"])
        if code == 0:
            result.usb_devices = [line.strip() for line in out.splitlines() if line.strip()]
            result.xtion_usb_visible = any(
                token in line.lower()
                for line in result.usb_devices
                for token in ("1d27:0600", "1d27:0601", "primesense", "xtion")
            )
        else:
            logging.debug("lsusb failed: %s", err)

    for name in ("OpenNI2", "openni2", "OpenNI", "oni"):
        lib = ctypes.util.find_library(name)
        if lib:
            result.openni_library = lib
            break

    try:
        from primesense import openni2 as _openni2  # noqa: F401

        result.primesense_available = True
    except Exception:
        result.primesense_available = False

    for attr, module_name in (
        ("opencv_available", "cv2"),
        ("open3d_available", "open3d"),
        ("trimesh_available", "trimesh"),
    ):
        try:
            __import__(module_name)
            setattr(result, attr, True)
        except Exception:
            setattr(result, attr, False)

    try:
        import torch

        result.torch_available = True
        result.torch_version = torch.__version__
        result.torch_cuda_available = bool(torch.cuda.is_available())
    except Exception:
        result.torch_available = False

    if check_bundle_cuda:
        add_bundlesdf_paths()
        for mod_name, attr in (("common", "bundle_cuda_common"), ("gridencoder", "bundle_cuda_gridencoder")):
            try:
                __import__(mod_name)
                setattr(result, attr, True)
            except Exception as exc:
                result.bundle_cuda_errors[mod_name] = f"{type(exc).__name__}: {exc}"

    return result


def print_preflight(result: PreflightResult) -> None:
    print("=" * 60)
    print("XTION / FOUNDATIONPOSE PREFLIGHT")
    print("=" * 60)
    print(f"USB bus visible: {result.dev_bus_usb_exists}")
    print(f"Xtion USB visible: {result.xtion_usb_visible}")
    for dev in result.usb_devices:
        print(f"USB: {dev}")
    print(f"Video devices: {result.video_devices if result.video_devices else 'none'}")
    print(f"OpenNI library: {result.openni_library or 'MISSING'}")
    print(f"primesense.openni2: {'OK' if result.primesense_available else 'MISSING'}")
    print(f"OpenCV: {'OK' if result.opencv_available else 'MISSING'}")
    print(f"Open3D: {'OK' if result.open3d_available else 'MISSING'}")
    print(f"trimesh: {'OK' if result.trimesh_available else 'MISSING'}")
    print(f"torch: {'OK ' + str(result.torch_version) if result.torch_available else 'MISSING'}")
    print(f"torch CUDA available: {result.torch_cuda_available}")
    print(f"BundleSDF common: {'OK' if result.bundle_cuda_common else 'MISSING'}")
    print(f"BundleSDF gridencoder: {'OK' if result.bundle_cuda_gridencoder else 'MISSING'}")
    for name, err in result.bundle_cuda_errors.items():
        print(f"  {name}: {err}")
    if not (result.bundle_cuda_common and result.bundle_cuda_gridencoder):
        print("BundleSDF CUDA ops rebuild command:")
        print("  source ./env_5070ti.sh && cd bundlesdf/mycuda && python -m pip install --no-build-isolation -e .")
    if result.xtion_usb_visible and not result.primesense_available:
        print("Xtion USB is visible, but OpenNI2/primesense is not available yet.")
    print("=" * 60)


class XtionOpenNI2Camera:
    def __init__(self, width: int, height: int, fps: int, redist: str | None = None) -> None:
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.redist = redist
        self.openni2: Any = None
        self.c_api: Any = None
        self.device: Any = None
        self.color_stream: Any = None
        self.depth_stream: Any = None
        self.info = CameraInfo()

    def __enter__(self) -> "XtionOpenNI2Camera":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def open(self) -> None:
        try:
            from primesense import _openni2 as c_api
            from primesense import openni2
        except Exception as exc:
            raise RuntimeError(
                "primesense.openni2 is unavailable. Xtion USB may be visible, "
                "but OpenNI2 runtime/Python bindings are required before capture."
            ) from exc

        self.openni2 = openni2
        self.c_api = c_api
        openni2.initialize(self.redist) if self.redist else openni2.initialize()

        try:
            self.device = openni2.Device.open_any()
            dev_info = self.device.get_device_info()
            self.info.name = getattr(dev_info, "name", "ASUS Xtion Pro Live")
            self.info.vendor = getattr(dev_info, "vendor", None)
            self.info.uri = getattr(dev_info, "uri", None)

            self.color_stream = self.device.create_color_stream()
            self.depth_stream = self.device.create_depth_stream()
            self._set_video_modes()
            self._enable_registration()
            self.color_stream.start()
            self.depth_stream.start()
            rgb, depth = self.read()
            self.info.rgb_resolution = (rgb.shape[1], rgb.shape[0])
            self.info.depth_resolution = (depth.shape[1], depth.shape[0])
            self.info.fps = self.requested_fps
            self.info.K, self.info.K_source = self.derive_intrinsics()
        except Exception:
            self.close()
            raise

    def _set_video_modes(self) -> None:
        assert self.c_api is not None
        modes = (
            (self.color_stream, self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888),
            (self.depth_stream, self.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM),
        )
        for stream, pix_fmt in modes:
            mode = self.c_api.OniVideoMode(
                pixelFormat=pix_fmt,
                resolutionX=self.requested_width,
                resolutionY=self.requested_height,
                fps=self.requested_fps,
            )
            try:
                stream.set_video_mode(mode)
            except Exception as exc:
                logging.warning("Requested %sx%s@%s mode was not accepted: %s", self.requested_width, self.requested_height, self.requested_fps, exc)

    def _enable_registration(self) -> None:
        assert self.openni2 is not None
        try:
            mode = self.openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
            if self.device.is_image_registration_mode_supported(mode):
                self.device.set_image_registration_mode(mode)
                self.info.depth_registered_to_rgb = True
            else:
                self.info.depth_registered_to_rgb = False
        except Exception as exc:
            logging.warning("Depth-to-color registration could not be enabled: %s", exc)
            self.info.depth_registered_to_rgb = False

    def derive_intrinsics(self) -> tuple[np.ndarray, str]:
        rgb, _ = self.read()
        height, width = rgb.shape[:2]
        try:
            hfov = float(self.color_stream.get_horizontal_fov())
            vfov = float(self.color_stream.get_vertical_fov())
            if hfov > 0 and vfov > 0:
                fx = width / (2.0 * math.tan(hfov / 2.0))
                fy = height / (2.0 * math.tan(vfov / 2.0))
                cx = (width - 1.0) / 2.0
                cy = (height - 1.0) / 2.0
                return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64), "OpenNI color stream FOV-derived"
        except Exception as exc:
            logging.debug("Could not derive K from OpenNI FOV: %s", exc)
        raise RuntimeError("OpenNI did not expose usable FOV intrinsics. Provide --K /path/to/K.txt.")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        color_frame = self.color_stream.read_frame()
        depth_frame = self.depth_stream.read_frame()
        cw, ch = color_frame.width, color_frame.height
        dw, dh = depth_frame.width, depth_frame.height
        rgb = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(ch, cw, 3).copy()
        depth = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(dh, dw).copy()
        if rgb.shape[:2] != depth.shape:
            raise RuntimeError(f"RGB/depth shape mismatch: rgb={rgb.shape[:2]} depth={depth.shape}; registration/alignment failed")
        return rgb, depth

    def close(self) -> None:
        for stream in (self.depth_stream, self.color_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
        self.depth_stream = None
        self.color_stream = None
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None
        if self.openni2 is not None:
            try:
                self.openni2.unload()
            except Exception:
                pass
            self.openni2 = None


def load_K(path: Path) -> np.ndarray:
    K = np.loadtxt(path).reshape(3, 3).astype(np.float64)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[2, 2] - 1) > 1e-6:
        raise ValueError(f"Invalid K matrix: {path}")
    return K


def validate_depth(depth_mm: np.ndarray, min_depth: float, max_depth: float) -> dict[str, float]:
    if depth_mm.dtype != np.uint16:
        raise RuntimeError(f"Depth dtype is {depth_mm.dtype}, expected uint16 millimetres")
    valid = depth_mm > 0
    valid &= depth_mm >= int(min_depth * 1000.0)
    valid &= depth_mm <= int(max_depth * 1000.0)
    ratio = float(valid.mean())
    if ratio < 0.02:
        raise RuntimeError(f"Too few valid depth pixels: {ratio:.4f}")
    vals = depth_mm[valid]
    stats = {
        "valid_ratio": ratio,
        "min_valid_mm": float(vals.min()),
        "median_valid_mm": float(np.median(vals)),
        "max_valid_mm": float(vals.max()),
    }
    print(
        "Depth validation: "
        f"dtype={depth_mm.dtype} min_valid={stats['min_valid_mm']:.0f} mm "
        f"median_valid={stats['median_valid_mm']:.0f} mm max_valid={stats['max_valid_mm']:.0f} mm "
        f"valid_ratio={ratio:.3f}"
    )
    return stats


def depth_preview(depth_mm: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    depth_m = depth_mm.astype(np.float32) / 1000.0
    vis = np.clip((depth_m - min_depth) / max(max_depth - min_depth, 1e-6), 0.0, 1.0)
    vis[depth_mm == 0] = 0.0
    vis8 = (255.0 * (1.0 - vis)).astype(np.uint8)
    return cv2.applyColorMap(vis8, cv2.COLORMAP_TURBO)


def grabcut_depth_mask(rgb: np.ndarray, depth_mm: np.ndarray, roi: tuple[int, int, int, int], min_depth: float, max_depth: float) -> np.ndarray:
    x, y, w, h = roi
    H, W = depth_mm.shape
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    roi_mask = np.zeros((H, W), np.uint8)
    roi_mask[y : y + h, x : x + w] = 1
    valid_depth = (depth_mm >= int(min_depth * 1000.0)) & (depth_mm <= int(max_depth * 1000.0))
    roi_depth = depth_mm[(roi_mask > 0) & valid_depth]
    if roi_depth.size < 50:
        raise RuntimeError("ROI has too few valid depth pixels for object mask initialization")
    med = float(np.median(roi_depth))
    mad = float(np.median(np.abs(roi_depth.astype(np.float32) - med)))
    band = max(80.0, 3.0 * mad)
    depth_seed = valid_depth & (np.abs(depth_mm.astype(np.float32) - med) <= band)

    seed = np.zeros((H, W), np.uint8)
    seed[(roi_mask > 0) & depth_seed] = 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    best = 0
    best_area = 0
    for idx in range(1, num):
        overlap = np.count_nonzero((labels == idx) & (roi_mask > 0))
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if overlap > 0 and area > best_area:
            best = idx
            best_area = area
    if best == 0:
        raise RuntimeError("Could not find a depth-connected foreground component in ROI")

    mask = np.zeros((H, W), np.uint8)
    mask[labels == best] = 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    gc_mask = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    gc_mask[roi_mask > 0] = cv2.GC_PR_FGD
    gc_mask[mask > 0] = cv2.GC_FGD
    gc_mask[~valid_depth] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb[..., ::-1], gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        refined &= depth_seed.astype(np.uint8) * 255
        if np.count_nonzero(refined) > 50:
            mask = refined
    except Exception as exc:
        logging.debug("GrabCut refinement failed, using depth mask: %s", exc)
    return mask.astype(np.uint8)


def validate_mask_depth(mask: np.ndarray, depth_mm: np.ndarray, min_depth: float, max_depth: float) -> dict[str, float]:
    fg = mask > 0
    if mask.dtype != np.uint8:
        raise RuntimeError(f"Mask dtype is {mask.dtype}, expected uint8")
    if np.count_nonzero(fg) < 100:
        raise RuntimeError("Mask is empty or too small")
    valid = fg & (depth_mm >= int(min_depth * 1000.0)) & (depth_mm <= int(max_depth * 1000.0))
    coverage = float(np.count_nonzero(valid)) / float(np.count_nonzero(fg))
    if coverage < 0.20:
        raise RuntimeError(f"Too little valid depth inside mask: {coverage:.3f}")
    vals = depth_mm[valid]
    depth_range_m = float(vals.max() - vals.min()) / 1000.0
    if depth_range_m > 1.0:
        raise RuntimeError(f"Object depth range is implausibly large: {depth_range_m:.3f} m")
    return {
        "mask_ratio": float(fg.mean()),
        "mask_depth_coverage": coverage,
        "object_depth_range_m": depth_range_m,
    }


def make_rgbd(rgb: np.ndarray, depth_mm: np.ndarray):
    import open3d as o3d

    # Open3D receives metres internally. Disk depth remains uint16 millimetres.
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth_m),
        depth_scale=1.0,
        depth_trunc=4.0,
        convert_rgb_to_intensity=False,
    )


def estimate_current_from_previous(prev_rgb: np.ndarray, prev_depth: np.ndarray, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray) -> tuple[bool, np.ndarray, str]:
    import open3d as o3d

    H, W = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    source = make_rgbd(prev_rgb, prev_depth)
    target = make_rgbd(rgb, depth)
    option = o3d.pipelines.odometry.OdometryOption()
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    success, current_from_previous, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        source,
        target,
        intrinsic,
        np.eye(4, dtype=np.float64),
        jacobian,
        option,
    )
    status = f"success={success} info_trace={float(np.trace(info)) if info is not None else math.nan:.3f}"
    return bool(success), np.asarray(current_from_previous, dtype=np.float64), status


def validate_pose(T: np.ndarray) -> None:
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise RuntimeError("Pose matrix is not finite 4x4")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5):
        raise RuntimeError("Pose bottom row is invalid")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=5e-2):
        raise RuntimeError("Pose rotation is not approximately orthonormal")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 5e-2:
        raise RuntimeError(f"Pose rotation determinant is not +1: {det:.4f}")


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1 @ R2.T
    val = (np.trace(R) - 1.0) / 2.0
    return float(math.acos(np.clip(val, -1.0, 1.0)))


def view_direction(frame: CandidateFrame, object_center: np.ndarray) -> np.ndarray:
    cam_pos = frame.cam_in_ob[:3, 3]
    direction = object_center - cam_pos
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        direction = -frame.cam_in_ob[:3, 2]
        norm = np.linalg.norm(direction)
    return direction / max(norm, 1e-9)


def masked_points_in_object(frame: CandidateFrame, K: np.ndarray, max_points: int = 2500) -> np.ndarray:
    mask = (frame.mask > 0) & (frame.depth_mm > 0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(np.int64)
        xs = xs[idx]
        ys = ys[idx]
    z = frame.depth_mm[ys, xs].astype(np.float64) / 1000.0
    x = (xs.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (ys.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    # cam_in_ob is camera_from_object, so invert it to express depth points in the common object frame.
    ob_in_cam = np.linalg.inv(frame.cam_in_ob)
    pts_ob = (ob_in_cam @ pts_cam.T).T[:, :3]
    return pts_ob[np.isfinite(pts_ob).all(axis=1)]


def select_diverse_views(frames: list[CandidateFrame], num_views: int, K: np.ndarray) -> list[CandidateFrame]:
    if len(frames) <= num_views:
        return frames
    pts = []
    for fr in frames:
        pts_ob = masked_points_in_object(fr, K)
        if len(pts_ob) > 0:
            pts.append(np.median(pts_ob, axis=0))
    object_center = np.median(np.asarray(pts), axis=0) if pts else np.zeros(3)
    selected = [max(frames, key=lambda fr: fr.quality_score)]
    while len(selected) < num_views:
        best = None
        best_score = -np.inf
        for fr in frames:
            if any(fr is chosen for chosen in selected):
                continue
            min_dist = np.inf
            for chosen in selected:
                rot = rotation_angle(fr.cam_in_ob[:3, :3], chosen.cam_in_ob[:3, :3]) / math.pi
                v1 = view_direction(fr, object_center)
                v2 = view_direction(chosen, object_center)
                view = math.acos(float(np.clip(np.dot(v1, v2), -1.0, 1.0))) / math.pi
                min_dist = min(min_dist, 0.5 * rot + 0.5 * view)
            score = min_dist + 0.05 * fr.quality_score
            if score > best_score:
                best = fr
                best_score = score
        assert best is not None
        selected.append(best)
    return selected


def object_base_dir(output_dir: Path, object_name: str) -> Path:
    return output_dir / object_name / "ref_views" / "ob_0000001"


def prepare_output_dirs(base: Path, overwrite: bool) -> None:
    if base.exists() and any(base.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output already exists: {base}. Pass --overwrite to replace this capture.")
        shutil.rmtree(base)
    for rel in ("rgb", "depth_enhanced", "mask", "cam_in_ob", "model", "diagnostics/mask_preview", "diagnostics/depth_preview"):
        (base / rel).mkdir(parents=True, exist_ok=True)


def write_reference_views(base: Path, frames: list[CandidateFrame], K: np.ndarray, camera_info: CameraInfo, args: argparse.Namespace) -> None:
    prepare_output_dirs(base, args.overwrite)
    for idx, fr in enumerate(frames):
        name = f"{idx:06d}"
        imageio.imwrite(base / "rgb" / f"{name}.png", fr.rgb)
        cv2.imwrite(str(base / "depth_enhanced" / f"{name}.png"), fr.depth_mm.astype(np.uint16))
        cv2.imwrite(str(base / "mask" / f"{name}.png"), fr.mask.astype(np.uint8))
        np.savetxt(base / "cam_in_ob" / f"{name}.txt", fr.cam_in_ob)
        overlay = fr.rgb.copy()
        overlay[fr.mask > 0] = (0.6 * overlay[fr.mask > 0] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
        imageio.imwrite(base / "diagnostics" / "mask_preview" / f"{name}.png", overlay)
        cv2.imwrite(str(base / "diagnostics" / "depth_preview" / f"{name}.png"), depth_preview(fr.depth_mm, args.min_depth, args.max_depth))
    np.savetxt(base / "K.txt", K)
    (base / "select_frames.yml").write_text("frames:\n" + "".join(f"  - {i}\n" for i in range(len(frames))), encoding="utf-8")
    write_metadata(base, frames, camera_info, args, model_file=None)
    make_contact_sheet(base, frames)


def write_metadata(base: Path, frames: list[CandidateFrame], camera_info: CameraInfo, args: argparse.Namespace, model_file: Path | None) -> None:
    object_root = base.parent.parent
    object_root.mkdir(parents=True, exist_ok=True)
    K_list = camera_info.K.tolist() if camera_info.K is not None else []
    metadata = {
        "object_name": args.object_name,
        "source": "ASUS Xtion Pro Live",
        "pipeline": "FoundationPose model-free",
        "num_reference_views": len(frames),
        "depth_unit_saved": "millimetres",
        "depth_dtype": "uint16",
        "rgb_resolution": list(camera_info.rgb_resolution or []),
        "depth_resolution": list(camera_info.depth_resolution or []),
        "depth_registered_to_rgb": bool(camera_info.depth_registered_to_rgb),
        "K": K_list,
        "K_source": camera_info.K_source,
        "model_file": str(model_file) if model_file else None,
        "coordinate_system_notes": (
            "cam_in_ob is OpenCV camera_from_object. The object frame is defined by the first accepted Xtion view. "
            "BundleSDF converts this to OpenGL internally with cam_in_ob @ diag(1,-1,-1,1). "
            "Xtion K is only for reconstruction; iPhone runtime must use iPhone K."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (object_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_contact_sheet(base: Path, frames: list[CandidateFrame]) -> None:
    if not frames:
        return
    thumbs = []
    for i, fr in enumerate(frames):
        img = fr.rgb.copy()
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{i:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        thumbs.append(img)
    cols = min(6, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 120, cols * 160, 3), np.uint8)
    for idx, img in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * 120 : (r + 1) * 120, c * 160 : (c + 1) * 160] = img
    imageio.imwrite(base / "diagnostics" / "selected_views.png", sheet)


def choose_roi(rgb: np.ndarray) -> tuple[int, int, int, int]:
    bgr = rgb[..., ::-1].copy()
    roi = cv2.selectROI("Select object ROI, then press ENTER/SPACE", bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select object ROI, then press ENTER/SPACE")
    if roi[2] <= 0 or roi[3] <= 0:
        raise RuntimeError("Object ROI selection was cancelled")
    return tuple(int(v) for v in roi)


def capture_reference_views(args: argparse.Namespace) -> tuple[Path, CameraInfo, list[CandidateFrame]]:
    result = preflight(check_bundle_cuda=True)
    print_preflight(result)
    if not result.xtion_usb_visible:
        raise RuntimeError("ASUS Xtion/PrimeSense USB device is not visible in WSL. Check usbipd/USB passthrough.")
    if not result.primesense_available:
        raise RuntimeError("Xtion USB is visible, but primesense.openni2 is missing. Install/configure OpenNI2 before capture.")

    base = object_base_dir(Path(args.output_dir).resolve(), args.object_name)
    if base.exists() and any(base.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Capture directory already exists: {base}. Use --overwrite only when you want to replace it.")

    candidates: list[CandidateFrame] = []
    roi: tuple[int, int, int, int] | None = None
    prev_rgb: np.ndarray | None = None
    prev_depth: np.ndarray | None = None
    cam_in_ob = np.eye(4, dtype=np.float64)
    last_accepted_pose: np.ndarray | None = None

    with XtionOpenNI2Camera(args.width, args.height, args.fps, args.openni2_redist) as cam:
        if args.K:
            cam.info.K = load_K(Path(args.K))
            cam.info.K_source = f"CLI override: {args.K}"
        assert cam.info.K is not None
        K = cam.info.K
        print(f"Camera: {cam.info.vendor or ''} {cam.info.name} {cam.info.uri or ''}".strip())
        print(f"RGB resolution: {cam.info.rgb_resolution}")
        print(f"Depth resolution: {cam.info.depth_resolution}")
        print(f"Depth-to-RGB registration: {cam.info.depth_registered_to_rgb}")
        print(f"K source: {cam.info.K_source}")
        print(K)

        print("Keys: SPACE=accept, M=reselect mask, R=reset, F=finish, Q/ESC=quit")
        while True:
            rgb, depth = cam.read()
            validate_depth(depth, args.min_depth, args.max_depth)
            if prev_rgb is not None and prev_depth is not None:
                success, delta, odo_status = estimate_current_from_previous(prev_rgb, prev_depth, rgb, depth, K)
                if success:
                    cam_in_ob = delta @ cam_in_ob
                    validate_pose(cam_in_ob)
                else:
                    logging.warning("RGB-D odometry failed: %s", odo_status)
            else:
                odo_status = "first frame"

            try:
                if roi is not None:
                    mask = grabcut_depth_mask(rgb, depth, roi, args.min_depth, args.max_depth)
                    mask_stats = validate_mask_depth(mask, depth, args.min_depth, args.max_depth)
                else:
                    mask = np.zeros(depth.shape, np.uint8)
                    mask_stats = {"mask_ratio": 0.0, "mask_depth_coverage": 0.0}
            except Exception as exc:
                mask = np.zeros(depth.shape, np.uint8)
                mask_stats = {"mask_ratio": 0.0, "mask_depth_coverage": 0.0}
                logging.debug("Current mask invalid: %s", exc)

            rgb_vis = rgb[..., ::-1].copy()
            if np.count_nonzero(mask) > 0:
                red = np.zeros_like(rgb_vis)
                red[:, :, 2] = 255
                rgb_vis = np.where(mask[..., None] > 0, (0.65 * rgb_vis + 0.35 * red).astype(np.uint8), rgb_vis)
            dep_vis = depth_preview(depth, args.min_depth, args.max_depth)
            vis = np.hstack([rgb_vis, dep_vis])
            delta_text = "n/a"
            if last_accepted_pose is not None:
                trans = np.linalg.norm(cam_in_ob[:3, 3] - last_accepted_pose[:3, 3])
                rot = math.degrees(rotation_angle(cam_in_ob[:3, :3], last_accepted_pose[:3, :3]))
                delta_text = f"dT={trans:.3f}m dR={rot:.1f}deg"
            cv2.putText(vis, f"accepted={len(candidates)} / target={args.num_views}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"odometry={odo_status[:60]}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(vis, f"view={delta_text} mask={mask_stats['mask_ratio']:.3f}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Xtion object builder: RGB | depth", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                raise KeyboardInterrupt("User quit")
            if key in (ord("m"), ord("M")) or roi is None:
                roi = choose_roi(rgb)
                mask = grabcut_depth_mask(rgb, depth, roi, args.min_depth, args.max_depth)
                validate_mask_depth(mask, depth, args.min_depth, args.max_depth)
            elif key in (ord("r"), ord("R")):
                candidates.clear()
                prev_rgb = None
                prev_depth = None
                cam_in_ob = np.eye(4, dtype=np.float64)
                last_accepted_pose = None
                logging.info("Capture reset")
            elif key == ord(" "):
                if roi is None:
                    logging.warning("Select object ROI with M before accepting frames")
                else:
                    mask = grabcut_depth_mask(rgb, depth, roi, args.min_depth, args.max_depth)
                    mask_stats = validate_mask_depth(mask, depth, args.min_depth, args.max_depth)
                    if last_accepted_pose is not None:
                        trans = float(np.linalg.norm(cam_in_ob[:3, 3] - last_accepted_pose[:3, 3]))
                        rot_deg = math.degrees(rotation_angle(cam_in_ob[:3, :3], last_accepted_pose[:3, :3]))
                        if trans < args.min_translation and rot_deg < args.min_view_angle:
                            logging.warning("Rejected near-duplicate view: dT=%.3fm dR=%.1fdeg", trans, rot_deg)
                            prev_rgb, prev_depth = rgb, depth
                            continue
                    quality = mask_stats["mask_depth_coverage"] + min(mask_stats["mask_ratio"] / 0.20, 1.0)
                    candidates.append(CandidateFrame(rgb.copy(), depth.copy(), mask.copy(), cam_in_ob.copy(), time.time(), quality))
                    last_accepted_pose = cam_in_ob.copy()
                    logging.info("Accepted candidate %d", len(candidates))
            elif key in (ord("f"), ord("F")):
                break

            prev_rgb, prev_depth = rgb, depth

        cv2.destroyAllWindows()
        selected = select_diverse_views(candidates, int(args.num_views), K)
        if len(selected) < int(args.num_views):
            logging.warning("Only %d usable views selected; requested %d", len(selected), int(args.num_views))
        if not selected:
            raise RuntimeError("No valid reference views were captured")
        write_reference_views(base, selected, K, cam.info, args)
        return base, cam.info, selected


def sample_rgbd(args: argparse.Namespace) -> None:
    result = preflight(check_bundle_cuda=True)
    print_preflight(result)
    with XtionOpenNI2Camera(args.width, args.height, args.fps, args.openni2_redist) as cam:
        if args.K:
            cam.info.K = load_K(Path(args.K))
            cam.info.K_source = f"CLI override: {args.K}"
        print(f"Camera: {cam.info.vendor or ''} {cam.info.name} {cam.info.uri or ''}".strip())
        for i in range(args.sample_count):
            rgb, depth = cam.read()
            stats = validate_depth(depth, args.min_depth, args.max_depth)
            print(f"sample={i} rgb_shape={rgb.shape} depth_shape={depth.shape} depth_dtype={depth.dtype} stats={stats}")
        print(f"registration={cam.info.depth_registered_to_rgb}")
        print(f"K_source={cam.info.K_source}")
        print(cam.info.K)


def load_reference_data(base: Path) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    K = np.loadtxt(base / "K.txt").reshape(3, 3)
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for rgb_file in sorted((base / "rgb").glob("*.png")):
        stem = rgb_file.stem
        rgb = imageio.imread(rgb_file)[..., :3]
        depth_raw = cv2.imread(str(base / "depth_enhanced" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(base / "mask" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(base / "cam_in_ob" / f"{stem}.txt").reshape(4, 4)
        if depth_raw is None or mask is None:
            raise RuntimeError(f"Missing depth/mask for frame {stem}")
        if depth_raw.dtype != np.uint16:
            raise RuntimeError(f"Depth frame {stem} dtype is {depth_raw.dtype}, expected uint16 millimetres")
        validate_pose(pose)
        # BundleSDF expects metres in memory. The PNG on disk remains uint16 mm.
        depths.append(depth_raw.astype(np.float32) / 1000.0)
        rgbs.append(rgb)
        masks.append(mask.astype(np.uint8))
        poses.append(pose)
    if not rgbs:
        raise RuntimeError(f"No reference RGB frames found in {base}")
    return K, rgbs, depths, masks, poses


def run_reconstruction(base: Path, args: argparse.Namespace) -> Path:
    result = preflight(check_bundle_cuda=True)
    print_preflight(result)
    missing = []
    if not result.open3d_available:
        missing.append("Open3D")
    if not result.trimesh_available:
        missing.append("trimesh")
    if not result.torch_available:
        missing.append("torch")
    if not result.torch_cuda_available:
        missing.append("torch CUDA device")
    if not result.bundle_cuda_common or not result.bundle_cuda_gridencoder:
        missing.append("BundleSDF CUDA ops common/gridencoder")
    if missing:
        raise RuntimeError("Cannot reconstruct until these dependencies are fixed: " + ", ".join(missing))

    add_bundlesdf_paths()
    from run_nerf import run_neural_object_field

    with open(BUNDLESDF_DIR / "config_ycbv.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if args.n_step is not None:
        cfg["n_step"] = int(args.n_step)

    save_dir = base / "nerf"
    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
        else:
            raise RuntimeError(f"Existing reconstruction directory found: {save_dir}. Pass --overwrite to replace it.")
    save_dir.mkdir(parents=True, exist_ok=True)

    K, rgbs, depths, masks, poses = load_reference_data(base)
    try:
        mesh = run_neural_object_field(cfg, K, rgbs, depths, masks, poses, save_dir=str(save_dir), debug=0)
    except Exception as exc:
        latest = save_dir / f"step_{int(cfg['n_step']):07d}_mesh_real_world.obj"
        if not latest.is_file():
            raise
        logging.warning("Final texture/export path failed, using fallback mesh %s: %s", latest, exc)
        import trimesh

        mesh = trimesh.load(str(latest), force="mesh")

    out_file = base / "model" / "model.obj"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_file)
    verify_mesh(out_file)
    frames_stub = [CandidateFrame(np.empty((0, 0, 3), np.uint8), np.empty((0, 0), np.uint16), np.empty((0, 0), np.uint8), np.eye(4), 0.0, 0.0) for _ in rgbs]
    camera_info = CameraInfo(K=K, K_source="loaded from existing capture", rgb_resolution=(rgbs[0].shape[1], rgbs[0].shape[0]), depth_resolution=(depths[0].shape[1], depths[0].shape[0]))
    write_metadata(base, frames_stub, camera_info, args, out_file)
    return out_file


def verify_mesh(model_file: Path) -> dict[str, Any]:
    import trimesh

    if not model_file.is_file():
        raise RuntimeError(f"Reconstructed mesh does not exist: {model_file}")
    mesh = trimesh.load(str(model_file), force="mesh")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(vertices) <= 0 or len(faces) <= 0:
        raise RuntimeError("Mesh has no vertices or faces")
    if not np.isfinite(vertices).all():
        raise RuntimeError("Mesh contains non-finite vertices")
    dims = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
    if not np.isfinite(dims).all() or np.any(dims <= 0):
        raise RuntimeError(f"Invalid mesh dimensions: {dims}")
    if np.any(dims < 0.001) or np.any(dims > 5.0):
        raise RuntimeError(f"Mesh scale is implausible for an Xtion object capture: {dims} metres")
    print("=" * 60)
    print("XTION MODEL-FREE OBJECT BUILD COMPLETE")
    print(f"Mesh vertices: {len(vertices):,}")
    print(f"Mesh faces: {len(faces):,}")
    print("Object dimensions:")
    print(f"X: {dims[0]:.3f} m")
    print(f"Y: {dims[1]:.3f} m")
    print(f"Z: {dims[2]:.3f} m")
    print(f"FoundationPose model: {model_file}")
    print("NEXT STEP:")
    print("Use this mesh with the existing iPhone FoundationPose tracking pipeline. Xtion is no longer required.")
    print("=" * 60)
    return {"vertices": len(vertices), "faces": len(faces), "dimensions_m": dims.tolist()}


def verify_saved_reference(base: Path) -> None:
    K, rgbs, depths, masks, poses = load_reference_data(base)
    if not np.isfinite(K).all():
        raise RuntimeError("Saved K is invalid")
    for i, (rgb, depth_m, mask, pose) in enumerate(zip(rgbs, depths, masks, poses)):
        if rgb.shape[:2] != depth_m.shape or mask.shape[:2] != depth_m.shape:
            raise RuntimeError(f"Saved frame {i} shape mismatch")
        if np.count_nonzero(depth_m > 0) == 0:
            raise RuntimeError(f"Saved frame {i} has no valid depth")
        if np.count_nonzero(mask > 0) == 0:
            raise RuntimeError(f"Saved frame {i} has empty mask")
        validate_pose(pose)
    print(f"Saved reference validation OK: {base} ({len(rgbs)} frames)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model-free FoundationPose object asset from an ASUS Xtion Pro Live RGB-D capture. "
            "Controls during capture: SPACE accept frame, M redefine mask ROI, R reset, F finish, Q/ESC quit."
        )
    )
    parser.add_argument("--object-name", default="xtion_object", help="Object/session name under --output-dir.")
    parser.add_argument("--output-dir", default="./xtion_models", help="Output root directory.")
    parser.add_argument("--num-views", type=int, default=24, help="Target number of diverse reference views.")
    parser.add_argument("--width", type=int, default=640, help="Requested RGB/depth width.")
    parser.add_argument("--height", type=int, default=480, help="Requested RGB/depth height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--K", default=None, help="Optional calibrated 3x3 Xtion intrinsics text file.")
    parser.add_argument("--openni2-redist", default=None, help="Optional OpenNI2 Redist directory for primesense.openni2.initialize().")
    parser.add_argument("--min-depth", type=float, default=0.25, help="Minimum valid depth in metres.")
    parser.add_argument("--max-depth", type=float, default=3.0, help="Maximum valid depth in metres.")
    parser.add_argument("--min-view-angle", type=float, default=8.0, help="Reject accepted views with less rotation than this unless translation is sufficient.")
    parser.add_argument("--min-translation", type=float, default=0.03, help="Reject accepted views with less translation than this unless rotation is sufficient.")
    parser.add_argument("--capture-only", action="store_true", help="Capture/write reference views without BundleSDF reconstruction.")
    parser.add_argument("--reconstruct-only", action="store_true", help="Reuse an existing capture and only run BundleSDF reconstruction.")
    parser.add_argument("--discover-only", action="store_true", help="Only print dependency/USB/OpenNI/CUDA discovery and exit.")
    parser.add_argument("--sample-rgbd", action="store_true", help="Open Xtion and capture validation samples without writing a full object capture.")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of frames for --sample-rgbd.")
    parser.add_argument("--n_step", type=int, default=None, help="Override BundleSDF config_ycbv.yml n_step for reconstruction.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing capture/reconstruction output.")
    parser.add_argument("--debug", action="store_true", help="Verbose logging.")
    args = parser.parse_args()
    if args.capture_only and args.reconstruct_only:
        parser.error("--capture-only and --reconstruct-only are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    setup_logging(args.debug)
    if args.discover_only:
        print_preflight(preflight(check_bundle_cuda=True))
        return 0
    if args.sample_rgbd:
        sample_rgbd(args)
        return 0

    base = object_base_dir(Path(args.output_dir).resolve(), args.object_name)
    camera_info: CameraInfo | None = None
    frames: list[CandidateFrame] = []
    try:
        if args.reconstruct_only:
            if not base.is_dir():
                raise RuntimeError(f"Existing capture directory not found: {base}")
        else:
            base, camera_info, frames = capture_reference_views(args)
            verify_saved_reference(base)
            if args.capture_only:
                print(f"Capture complete: {base}")
                print(f"Later reconstruction command: python run_xtion_object_builder.py --object-name {args.object_name} --output-dir {args.output_dir} --reconstruct-only")
                return 0

        model_file = run_reconstruction(base, args)
        if camera_info is not None:
            write_metadata(base, frames, camera_info, args, model_file)
        print(f"Final model.obj path: {model_file}")
        print("iPhone tracking stage: pass this path as --mesh_file to your existing run_live.py command; keep using iPhone RGB/LiDAR K at runtime.")
        return 0
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        print(f"Interrupted: {exc}")
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
