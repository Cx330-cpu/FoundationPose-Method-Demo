from datareader import *
import argparse
import os
import socket
import time

from network_frame_protocol import send_frame


def timestamp_from_frame_id(frame_id):
  try:
    return int(frame_id) * 1e-9
  except ValueError:
    return None


def send_recorded_frames(args):
  set_logging_format()
  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  frame_interval = 0 if args.fps <= 0 else 1.0 / args.fps

  with socket.create_connection((args.host, args.port), timeout=args.connect_timeout) as sock:
    logging.info(f"connected to {args.host}:{args.port}")
    for i in range(len(reader.color_files)):
      frame_start = time.perf_counter()
      frame_id = reader.id_strs[i]
      rgb = reader.get_color(i)
      depth = reader.get_depth(i)
      mask = reader.get_mask(0).astype(bool) if i == 0 else None
      send_frame(
        sock=sock,
        rgb=rgb,
        depth=depth,
        K=reader.K,
        mask=mask,
        frame_id=frame_id,
        index=i,
        timestamp=timestamp_from_frame_id(frame_id),
        rgb_codec=args.rgb_codec,
        jpeg_quality=args.jpeg_quality,
      )
      logging.info(f"sent frame {i}/{len(reader.color_files)} id:{frame_id}")

      if frame_interval > 0:
        elapsed = time.perf_counter() - frame_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
          time.sleep(sleep_time)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--host', type=str, default='127.0.0.1')
  parser.add_argument('--port', type=int, default=5000)
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--rgb_codec', choices=['png', 'jpeg'], default='png')
  parser.add_argument('--jpeg_quality', type=int, default=95)
  parser.add_argument('--fps', type=float, default=0)
  parser.add_argument('--connect_timeout', type=float, default=30)
  args = parser.parse_args()
  send_recorded_frames(args)
