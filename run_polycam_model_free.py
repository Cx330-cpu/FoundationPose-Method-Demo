#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

from tools.polycam_to_foundationpose import convert


def main():
    parser = argparse.ArgumentParser(description="Polycam Raw Data -> FoundationPose model-free reconstructed model.obj")
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
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--convert_only", action="store_true")
    parser.add_argument("--prepare-only", dest="prepare_only", action="store_true")
    parser.add_argument("--quality-check-only", dest="quality_check_only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n_step", type=int, default=None)
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
    args = parser.parse_args()

    ob_dir = convert(args)
    if args.validate_only or args.convert_only or args.prepare_only or args.quality_check_only or not args.run:
        print(f"Converted FoundationPose object directory: {ob_dir}")
        return

    cmd = [sys.executable, "bundlesdf/run_nerf_custom.py", "--base_dir", str(ob_dir), "--force"]
    if args.n_step is not None:
        cmd += ["--n_step", str(args.n_step)]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
