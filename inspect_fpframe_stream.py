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
  x = np.arange(width, dtype=np.int32)
  row = (700 + (x * 100 // max(1, width - 1))).astype(np.uint16)
  return np.tile(row[None, :], (height, 1))


def build_synthetic_mask(width, height):
  mask = np.zeros((height, width), dtype=bool)
  x_min = width // 4
  x_max = width * 3 // 4
  y_min = height // 4
  y_max = height * 3 // 4
  mask[y_min:y_max, x_min:x_max] = True
  return mask


def compare_synthetic(decoded):
  header = decoded["header"]
  width = header["width"]
  height = header["height"]
  index = header["index"]

  expected_rgb = build_synthetic_rgb(width, height, index)
  rgb_diff = np.abs(decoded["rgb"].astype(np.int16) - expected_rgb.astype(np.int16))

  decoded_depth_mm = np.rint(decoded["depth"] * 1000.0).astype(np.uint16)
  expected_depth_mm = build_synthetic_depth_mm(width, height)
  depth_diff = np.abs(decoded_depth_mm.astype(np.int32) - expected_depth_mm.astype(np.int32))

  expected_K = np.array([
    [width * 0.9, 0.0, width * 0.5],
    [0.0, width * 0.9, height * 0.5],
    [0.0, 0.0, 1.0],
  ])
  K_diff = np.abs(decoded["K"] - expected_K)

  if index == 0:
    expected_mask = build_synthetic_mask(width, height)
    mask_exact = decoded["mask"] is not None and np.array_equal(decoded["mask"], expected_mask)
  else:
    mask_exact = decoded["mask"] is None

  return {
    "rgb_exact": bool(np.array_equal(decoded["rgb"], expected_rgb)),
    "rgb_max_diff": int(rgb_diff.max()),
    "depth_exact": bool(np.array_equal(decoded_depth_mm, expected_depth_mm)),
    "depth_max_diff_mm": int(depth_diff.max()),
    "K_exact": bool(np.array_equal(decoded["K"], expected_K)),
    "K_max_diff": float(K_diff.max()),
    "mask_exact": bool(mask_exact),
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
