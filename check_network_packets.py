from datareader import *
import argparse
import os
import socket
import subprocess
import sys

from network_frame_protocol import receive_frame


def max_abs_diff(a, b):
  return float(np.max(np.abs(a - b)))


def check_network_packets(args):
  set_logging_format()
  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server_sock.bind((args.host, args.port))
  server_sock.listen(1)

  sender_cmd = [
    sys.executable,
    args.sender_script,
    '--host', args.connect_host,
    '--port', str(args.port),
    '--test_scene_dir', args.test_scene_dir,
    '--rgb_codec', args.rgb_codec,
    '--jpeg_quality', str(args.jpeg_quality),
    '--fps', '0',
  ]
  sender = subprocess.Popen(sender_cmd)

  rgb_exact = True
  depth_exact = True
  k_exact = True
  mask_exact = True
  rgb_max = 0
  depth_max = 0.0
  k_max = 0.0

  try:
    client_sock, client_addr = server_sock.accept()
    logging.info(f"accepted packet checker sender from {client_addr}")
    with client_sock:
      for i in range(len(reader.color_files)):
        decoded = receive_frame(client_sock)
        header = decoded["header"]
        rgb = decoded["rgb"]
        depth = decoded["depth"]
        K = decoded["K"]
        mask = decoded["mask"]

        expected_rgb = reader.get_color(i)
        expected_depth = reader.get_depth(i)
        expected_K = reader.K
        expected_mask = reader.get_mask(0).astype(bool) if i == 0 else None

        rgb_diff = int(np.max(np.abs(rgb.astype(np.int16) - expected_rgb.astype(np.int16))))
        depth_diff = max_abs_diff(depth, expected_depth)
        k_diff = max_abs_diff(K, expected_K)
        if expected_mask is None:
          cur_mask_exact = mask is None
        else:
          cur_mask_exact = mask is not None and np.array_equal(mask, expected_mask)

        rgb_max = max(rgb_max, rgb_diff)
        depth_max = max(depth_max, depth_diff)
        k_max = max(k_max, k_diff)
        rgb_exact = rgb_exact and np.array_equal(rgb, expected_rgb)
        depth_exact = depth_exact and np.array_equal(depth, expected_depth)
        k_exact = k_exact and np.array_equal(K, expected_K)
        mask_exact = mask_exact and cur_mask_exact

        if header["frame_id"] != reader.id_strs[i] or header["index"] != i:
          raise RuntimeError(f"header frame mismatch at {i}: {header}")

    sender_ret = sender.wait(timeout=args.sender_timeout)
    if sender_ret != 0:
      raise RuntimeError(f"sender exited with code {sender_ret}")
  finally:
    server_sock.close()
    if sender.poll() is None:
      sender.terminate()

  print(f"num_packets: {len(reader.color_files)}")
  print(f"RGB exact: {rgb_exact}")
  print(f"RGB max diff: {rgb_max}")
  print(f"Depth exact: {depth_exact}")
  print(f"Depth max diff: {depth_max:.12g}")
  print(f"K exact: {k_exact}")
  print(f"K max diff: {k_max:.12g}")
  print(f"Mask exact: {mask_exact}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--host', type=str, default='0.0.0.0')
  parser.add_argument('--connect_host', type=str, default='127.0.0.1')
  parser.add_argument('--port', type=int, default=5001)
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--sender_script', type=str, default=f'{code_dir}/send_recorded_frames.py')
  parser.add_argument('--rgb_codec', choices=['png', 'jpeg'], default='png')
  parser.add_argument('--jpeg_quality', type=int, default=95)
  parser.add_argument('--sender_timeout', type=float, default=30)
  args = parser.parse_args()
  check_network_packets(args)
