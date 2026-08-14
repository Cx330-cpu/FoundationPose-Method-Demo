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


def save_packet(decoded, save_dir):
  os.makedirs(save_dir, exist_ok=True)
  frame_id = decoded["header"]["frame_id"]
  cv2.imwrite(os.path.join(save_dir, f"{frame_id}_rgb.png"), cv2.cvtColor(decoded["rgb"], cv2.COLOR_RGB2BGR))
  depth_mm = np.rint(decoded["depth"] * 1000.0).astype(np.uint16)
  cv2.imwrite(os.path.join(save_dir, f"{frame_id}_depth_mm.png"), depth_mm)
  if decoded["mask"] is not None:
    cv2.imwrite(os.path.join(save_dir, f"{frame_id}_mask.png"), decoded["mask"].astype(np.uint8) * 255)
  np.savetxt(os.path.join(save_dir, f"{frame_id}_K.txt"), decoded["K"])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=5000)
  parser.add_argument("--num_frames", type=int, default=1)
  parser.add_argument("--save_dir", default="debug_fpframe_inspect")
  parser.add_argument("--expect_synthetic", action="store_true")
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
      print(f"frame_id={header['frame_id']} index={header['index']} size={header['width']}x{header['height']} rgb_format={header['rgb_format']} mask_format={header['mask_format']}")
      print(f"K=\n{decoded['K']}")
      depth_mm = np.rint(decoded["depth"] * 1000.0).astype(np.uint16)
      print(f"depth_mm min={depth_mm.min()} max={depth_mm.max()} invalid={(depth_mm == 0).sum()}")
      print(f"mask_present={decoded['mask'] is not None}")
      save_packet(decoded, args.save_dir)

      if args.expect_synthetic:
        stats = compare_synthetic(decoded)
        print(f"synthetic_check={stats}")
  finally:
    client_sock.close()
    server_sock.close()


if __name__ == "__main__":
  main()
