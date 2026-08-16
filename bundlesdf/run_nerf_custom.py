#!/usr/bin/env python3
import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh
import yaml

code_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(code_dir.parent))
mycuda_dir = code_dir / "mycuda"
mycuda_build_dir = mycuda_dir / "build" / "lib.linux-x86_64-cpython-310"
for extension_dir in [mycuda_dir, mycuda_build_dir]:
    if extension_dir.is_dir():
        sys.path.insert(0, str(extension_dir))

from run_nerf import run_neural_object_field


def load_ob(base_dir):
    base = Path(base_dir)
    K = np.loadtxt(base / "K.txt")
    rgbs, depths, masks, poses = [], [], [], []
    for rgb_file in sorted((base / "rgb").glob("*.png")):
        stem = rgb_file.stem
        rgbs.append(imageio.imread(rgb_file)[..., :3])
        dep = cv2.imread(str(base / "depth_enhanced" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(base / "mask" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(base / "cam_in_ob" / f"{stem}.txt").reshape(4, 4)
        if dep is None or mask is None:
            raise RuntimeError(f"missing depth/mask for frame {stem}")
        depths.append(dep.astype(np.float32) / 1000.0)
        masks.append(mask.astype(np.uint8))
        poses.append(pose)
    if not rgbs:
        raise RuntimeError(f"no rgb frames found in {base}")
    return K, rgbs, depths, masks, poses


def main():
    parser = argparse.ArgumentParser(description="Run FoundationPose model-free Neural Object Field for one custom object.")
    parser.add_argument("--base_dir", required=True, help="Path to ob_0000001 containing rgb/depth_enhanced/mask/cam_in_ob/K.txt")
    parser.add_argument("--config", default=str(code_dir / "config_ycbv.yml"))
    parser.add_argument("--n_step", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_texture_fallback", action="store_true")
    parser.add_argument(
        "--disable_octree",
        action="store_true",
        help="Debug-only fallback. By default custom model-free training keeps the original BundleSDF octree settings.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if args.n_step is not None:
        cfg["n_step"] = args.n_step

    if args.disable_octree:
        cfg["use_octree"] = 0
        cfg["denoise_depth_use_octree_cloud"] = False
        cfg["save_octree_clouds"] = False

    save_dir = base / "nerf"
    if save_dir.exists() and args.force:
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    K, rgbs, depths, masks, poses = load_ob(base)
    try:
        mesh = run_neural_object_field(cfg, K, rgbs, depths, masks, poses, save_dir=str(save_dir), debug=0)
    except Exception as exc:
        latest = save_dir / f"step_{int(cfg['n_step']):07d}_mesh_real_world.obj"
        if args.no_texture_fallback or not latest.is_file():
            raise
        print(f"[run_nerf_custom] optional final texture/export path failed: {type(exc).__name__}: {exc}")
        print(f"[run_nerf_custom] using fallback mesh {latest}")
        mesh = trimesh.load(str(latest), force="mesh")

    out_file = base / "model" / "model.obj"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_file)
    mesh2 = trimesh.load(str(out_file), force="mesh")
    bounds = mesh2.bounds
    if len(mesh2.vertices) == 0 or len(mesh2.faces) == 0 or not np.isfinite(mesh2.vertices).all():
        raise RuntimeError(f"bad reconstructed mesh: {out_file}")
    print("MODEL FREE RECONSTRUCTION SUCCESS")
    print(f"model path: {out_file}")
    print(f"vertices: {len(mesh2.vertices)}")
    print(f"faces: {len(mesh2.faces)}")
    print(f"bounds: {bounds.tolist()}")


if __name__ == "__main__":
    main()
