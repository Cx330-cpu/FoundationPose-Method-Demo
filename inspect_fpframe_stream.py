import argparse
import os
import socket

import cv2
import numpy as np

from network_frame_protocol import StreamEnd, receive_frame


def build_synthetic_rgb(width, height, index):
  y, x = np.indices((height, width))
  rgb = np.empty((height, width, 3), dtype=np.uint8)
  rgb[..., 0] = (x + index * 3) % 256
  rgb[..., 1] = (y + index * 5) % 256
  rgb[..., 2] = ((x + y) // 2 + index * 7) % 256
  return rgb


def build_synthetic_depth_mm(width, height):
  y, x = np.indices((height, width), dtype=np.int32)
  depth = (
    700
    + (x * 100 // max(1, width - 1))
    + (y * 50 // max(1, height - 1))
  )
  return depth.astype(np.uint16)


def build_synthetic_mask(width, height):
  mask = np.zeros((height, width), dtype=bool)
  x_min = width // 8
  x_max = width // 2
  y_min = height // 6
  y_max = height // 2
  mask[y_min:y_max, x_min:x_max] = True

  notch_x_min = width // 8
  notch_x_max = width // 4
  notch_y_min = height // 6
  notch_y_max = height // 3
  mask[notch_y_min:notch_y_max, notch_x_min:notch_x_max] = False

  tab_x_min = width * 5 // 8
  tab_x_max = width * 3 // 4
  tab_y_min = height * 2 // 3
  tab_y_max = height * 5 // 6
  mask[tab_y_min:tab_y_max, tab_x_min:tab_x_max] = True
  return mask


def orientation_candidates(image):
  return {
    "identity": image,
    "flipud": np.flipud(image),
    "fliplr": np.fliplr(image),
    "flipud_fliplr": np.flipud(np.fliplr(image)),
  }


def compare_array_candidates(decoded_array, expected_array):
  stats = {}
  for name, candidate in orientation_candidates(decoded_array).items():
    diff = np.abs(candidate.astype(np.int64) - expected_array.astype(np.int64))
    stats[name] = {
      "exact": bool(np.array_equal(candidate, expected_array)),
      "max_diff": int(diff.max()) if diff.size else 0,
    }
  return stats


def compare_mask_candidates(decoded_mask, expected_mask):
  if decoded_mask is None:
    return {
      name: {
        "exact": False,
        "max_diff": 1,
        "mismatch_pixels": int(expected_mask.size),
      }
      for name in ["identity", "flipud", "fliplr", "flipud_fliplr"]
    }

  stats = {}
  for name, candidate in orientation_candidates(decoded_mask).items():
    mismatch = np.logical_xor(candidate.astype(bool), expected_mask)
    stats[name] = {
      "exact": bool(not mismatch.any()),
      "max_diff": int(mismatch.max()) if mismatch.size else 0,
      "mismatch_pixels": int(mismatch.sum()),
    }
  return stats


def compare_synthetic(decoded):
  header = decoded["header"]
  width = header["width"]
  height = header["height"]
  index = header["index"]

  expected_rgb = build_synthetic_rgb(width, height, index)
  rgb_candidates = compare_array_candidates(decoded["rgb"], expected_rgb)
  rgb_bgr_candidates = compare_array_candidates(decoded["rgb"][..., ::-1], expected_rgb)

  decoded_depth_mm = np.rint(decoded["depth"] * 1000.0).astype(np.uint16)
  expected_depth_mm = build_synthetic_depth_mm(width, height)
  depth_candidates = compare_array_candidates(decoded_depth_mm, expected_depth_mm)

  expected_K = np.array([
    [width * 0.9, 0.0, width * 0.5],
    [0.0, width * 0.9, height * 0.5],
    [0.0, 0.0, 1.0],
  ])
  K_diff = np.abs(decoded["K"] - expected_K)

  if index == 0:
    expected_mask = build_synthetic_mask(width, height)
    mask_candidates = compare_mask_candidates(decoded["mask"], expected_mask)
  else:
    mask_candidates = {
      "expected_none": {
        "exact": decoded["mask"] is None,
        "max_diff": 0 if decoded["mask"] is None else 1,
        "mismatch_pixels": 0 if decoded["mask"] is None else int(decoded["mask"].size),
      }
    }

  return {
    "rgb": rgb_candidates,
    "rgb_bgr_swap": rgb_bgr_candidates,
    "depth_mm": depth_candidates,
    "K_exact": bool(np.array_equal(decoded["K"], expected_K)),
    "K_max_diff": float(K_diff.max()),
    "mask": mask_candidates,
  }


def mask_bounds(mask):
  if mask is None:
    return None

  ys, xs = np.where(mask > 0)
  if xs.size == 0:
    return None

  x_min = int(xs.min())
  y_min = int(ys.min())
  x_max = int(xs.max())
  y_max = int(ys.max())
  width_px = x_max - x_min + 1
  height_px = y_max - y_min + 1
  center_x = x_min + (width_px - 1) * 0.5
  center_y = y_min + (height_px - 1) * 0.5
  return {
    "x_min": x_min,
    "y_min": y_min,
    "x_max": x_max,
    "y_max": y_max,
    "width_px": width_px,
    "height_px": height_px,
    "center_x": center_x,
    "center_y": center_y,
  }


def print_geometry_trace(decoded):
  header = decoded["header"]
  rgb = decoded["rgb"]
  depth = decoded["depth"]
  mask = decoded["mask"]
  K = decoded["K"]
  width = header["width"]
  height = header["height"]

  print("[FP-PC][FRAME]")
  print(f"frame_id={header['frame_id']}")
  print(f"index={header['index']}")
  print(f"timestamp={header['timestamp']}")
  print(f"size={width}x{height}")
  print(f"rgb={header['rgb_format']}")
  print(f"depth={header['depth_format']}")
  print(f"mask={header['mask_format']}")
  print(f"rgb_len={header['rgb_len']}")
  print(f"depth_len={header['depth_len']}")
  print(f"mask_len={header['mask_len']}")
  print(f"fx={K[0, 0]:.6f}")
  print(f"fy={K[1, 1]:.6f}")
  print(f"cx={K[0, 2]:.6f}")
  print(f"cy={K[1, 2]:.6f}")

  valid_depth = depth[np.isfinite(depth) & (depth > 0)]
  depth_min = float(valid_depth.min()) if valid_depth.size else float("nan")
  depth_max = float(valid_depth.max()) if valid_depth.size else float("nan")
  invalid_depth_count = int((~np.isfinite(depth) | (depth <= 0)).sum())
  print("[FP-PC][DECODE]")
  print(f"rgb={rgb.shape} {rgb.dtype}")
  print(f"depth={depth.shape} {depth.dtype}")
  if mask is None:
    print("mask=None maskPresent=False")
  else:
    print(f"mask={mask.shape} {mask.dtype} maskPresent=True")
  print(f"depthMinMeters={depth_min:.6f}")
  print(f"depthMaxMeters={depth_max:.6f}")
  print(f"invalidDepthCount={invalid_depth_count}")

  bounds = mask_bounds(mask)
  if mask is not None:
    print("[FP-PC][MASK]")
    print(f"frame_id={header['frame_id']}")
    print(f"timestamp={header['timestamp']}")
    print(f"imageWidth={width}")
    print(f"imageHeight={height}")
    print("pixelConvention=xMax/yMax are inclusive pixel indices; normalized width/height use inclusive pixel count")
    if bounds is None:
      print("emptyMask=True")
    else:
      norm_x = bounds["x_min"] / width
      norm_y = bounds["y_min"] / height
      norm_w = bounds["width_px"] / width
      norm_h = bounds["height_px"] / height
      print(f"xMin={bounds['x_min']}")
      print(f"yMin={bounds['y_min']}")
      print(f"xMax={bounds['x_max']}")
      print(f"yMax={bounds['y_max']}")
      print(f"widthPx={bounds['width_px']}")
      print(f"heightPx={bounds['height_px']}")
      print(f"centerPx=({bounds['center_x']:.3f},{bounds['center_y']:.3f})")
      print(f"normalizedTopLeft=({norm_x:.6f},{norm_y:.6f},{norm_w:.6f},{norm_h:.6f})")

      print("[FP-PC][MASK-CORNERS]")
      print("origin=TopLeft")
      print(f"left={bounds['x_min'] / width:.6f}")
      print(f"top={bounds['y_min'] / height:.6f}")
      print(f"right={(bounds['x_max'] + 1) / width:.6f}")
      print(f"bottom={(bounds['y_max'] + 1) / height:.6f}")

  if header["index"] == 0 and mask is not None:
    print("[FP-PC][REGISTER-CANDIDATE]")
    print(f"frame_id={header['frame_id']}")
    print(f"timestamp={header['timestamp']}")
    print(f"size={width}x{height}")
    if bounds is None:
      print("emptyMask=True")
    else:
      print(
        "maskBounds="
        f"(xMin={bounds['x_min']},yMin={bounds['y_min']},"
        f"xMax={bounds['x_max']},yMax={bounds['y_max']})"
      )
      print(
        "normalizedTopLeft="
        f"({bounds['x_min'] / width:.6f},{bounds['y_min'] / height:.6f},"
        f"{bounds['width_px'] / width:.6f},{bounds['height_px'] / height:.6f})"
      )
    print(f"K=({K[0, 0]:.6f},{K[1, 1]:.6f},{K[0, 2]:.6f},{K[1, 2]:.6f})")


def save_mask_overlay(decoded, save_dir):
  mask = decoded["mask"]
  bounds = mask_bounds(mask)
  if bounds is None:
    return None

  frame_id = decoded["header"]["frame_id"]
  overlay_rgb = decoded["rgb"].copy()
  overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
  cv2.rectangle(
    overlay_bgr,
    (bounds["x_min"], bounds["y_min"]),
    (bounds["x_max"], bounds["y_max"]),
    (0, 0, 255),
    2,
  )
  path = os.path.join(save_dir, f"{frame_id}_mask_overlay.png")
  cv2.imwrite(path, overlay_bgr)
  return path


def save_packet(decoded, save_dir, trace_geometry=False):
  os.makedirs(save_dir, exist_ok=True)
  frame_id = decoded["header"]["frame_id"]
  rgb_path = os.path.join(save_dir, f"{frame_id}_rgb.png")
  depth_path = os.path.join(save_dir, f"{frame_id}_depth_mm.png")
  mask_path = os.path.join(save_dir, f"{frame_id}_mask.png") if decoded["mask"] is not None else None
  K_path = os.path.join(save_dir, f"{frame_id}_K.txt")

  cv2.imwrite(rgb_path, cv2.cvtColor(decoded["rgb"], cv2.COLOR_RGB2BGR))
  depth_mm = np.rint(decoded["depth"] * 1000.0).astype(np.uint16)
  cv2.imwrite(depth_path, depth_mm)
  if decoded["mask"] is not None:
    cv2.imwrite(mask_path, decoded["mask"].astype(np.uint8) * 255)
  np.savetxt(K_path, decoded["K"])

  overlay_path = save_mask_overlay(decoded, save_dir) if trace_geometry else None
  if trace_geometry:
    print("[FP-PC][SAVE]")
    print(f"frame_id={frame_id}")
    print(f"rgb_path={rgb_path}")
    print(f"mask_path={mask_path}")
    print(f"depth_path={depth_path}")
    print(f"K_path={K_path}")
    if overlay_path is not None:
      print(f"overlay_path={overlay_path}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=5000)
  parser.add_argument("--num_frames", type=int, default=1)
  parser.add_argument("--save_dir", default="debug_fpframe_inspect")
  parser.add_argument("--expect_synthetic", action="store_true")
  parser.add_argument("--trace_geometry", action="store_true")
  args = parser.parse_args()

  server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server_sock.bind((args.host, args.port))
  server_sock.listen(1)
  print(f"waiting for FPFRAME sender on {args.host}:{args.port}")
  client_sock, client_addr = server_sock.accept()
  print(f"accepted sender from {client_addr}")

  try:
    for _ in range(args.num_frames):
      try:
        decoded = receive_frame(client_sock)
      except StreamEnd:
        break

      header = decoded["header"]
      if args.trace_geometry:
        print_geometry_trace(decoded)
      else:
        print(f"frame_id={header['frame_id']} index={header['index']} size={header['width']}x{header['height']} rgb_format={header['rgb_format']} mask_format={header['mask_format']}")
        print(f"K=\n{decoded['K']}")
        depth_mm = np.rint(decoded["depth"] * 1000.0).astype(np.uint16)
        print(f"depth_mm min={depth_mm.min()} max={depth_mm.max()} invalid={(depth_mm == 0).sum()}")
        print(f"mask_present={decoded['mask'] is not None}")
      save_packet(decoded, args.save_dir, trace_geometry=args.trace_geometry)

      if args.expect_synthetic:
        stats = compare_synthetic(decoded)
        print(f"synthetic_check={stats}")
  finally:
    client_sock.close()
    server_sock.close()


if __name__ == "__main__":
  main()
