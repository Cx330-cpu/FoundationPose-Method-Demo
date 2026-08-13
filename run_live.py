# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time
import shutil
import socket
from network_frame_protocol import StreamEnd, receive_frame


@dataclass
class FramePacket:
  rgb: np.ndarray
  depth: np.ndarray
  K: np.ndarray
  mask: Optional[np.ndarray]
  timestamp: Optional[float]
  frame_id: str
  index: int


class FrameProvider(ABC):
  @abstractmethod
  def get_frame(self) -> Optional[FramePacket]:
    pass


class RecordedFoundationPoseProvider(FrameProvider):
  def __init__(self, video_dir, shorter_side=None, zfar=np.inf):
    self.reader = YcbineoatReader(video_dir=video_dir, shorter_side=shorter_side, zfar=zfar)
    self.index = 0

  def __len__(self):
    return len(self.reader.color_files)

  def _timestamp_from_frame_id(self, frame_id):
    try:
      return int(frame_id) * 1e-9
    except ValueError:
      return None

  def get_frame(self) -> Optional[FramePacket]:
    if self.index >= len(self.reader.color_files):
      return None

    i = self.index
    frame_id = self.reader.id_strs[i]
    rgb = self.reader.get_color(i)
    depth = self.reader.get_depth(i)
    mask = self.reader.get_mask(0).astype(bool) if i == 0 else None
    packet = FramePacket(
      rgb=rgb,
      depth=depth,
      K=self.reader.K,
      mask=mask,
      timestamp=self._timestamp_from_frame_id(frame_id),
      frame_id=frame_id,
      index=i,
    )
    self.index += 1
    return packet


class NetworkFrameProvider(FrameProvider):
  def __init__(self, host, port):
    self.host = host
    self.port = port
    self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.server_sock.bind((host, port))
    self.server_sock.listen(1)
    logging.info(f"waiting for frame sender on {host}:{port}")
    self.client_sock, self.client_addr = self.server_sock.accept()
    logging.info(f"accepted frame sender from {self.client_addr}")

  def get_frame(self) -> Optional[FramePacket]:
    try:
      decoded = receive_frame(self.client_sock)
    except StreamEnd:
      return None

    header = decoded["header"]
    return FramePacket(
      rgb=decoded["rgb"],
      depth=decoded["depth"],
      K=decoded["K"],
      mask=decoded["mask"],
      timestamp=header["timestamp"],
      frame_id=header["frame_id"],
      index=header["index"],
    )

  def close(self):
    for sock in [getattr(self, "client_sock", None), getattr(self, "server_sock", None)]:
      if sock is not None:
        sock.close()


def validate_frame_packet(packet: FramePacket):
  if packet.rgb.ndim != 3 or packet.rgb.shape[2] != 3:
    raise ValueError(f"rgb must have shape (H, W, 3), got {packet.rgb.shape}")
  if packet.rgb.dtype != np.uint8:
    raise ValueError(f"rgb must be np.uint8, got {packet.rgb.dtype}")
  if packet.depth.ndim != 2:
    raise ValueError(f"depth must have shape (H, W), got {packet.depth.shape}")
  if packet.depth.shape != packet.rgb.shape[:2]:
    raise ValueError(f"depth shape {packet.depth.shape} does not match rgb shape {packet.rgb.shape[:2]}")
  if packet.K.shape != (3, 3):
    raise ValueError(f"K must have shape (3, 3), got {packet.K.shape}")
  if packet.mask is not None and packet.mask.shape != packet.depth.shape:
    raise ValueError(f"mask shape {packet.mask.shape} does not match depth shape {packet.depth.shape}")


def draw_live_overlay(packet: FramePacket, pose, to_origin, bbox, state, operation, fps):
  center_pose = pose @ np.linalg.inv(to_origin)
  vis = packet.rgb.copy()
  vis = draw_posed_3d_box(packet.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
  vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=packet.K, thickness=3, transparency=0, is_input_rgb=True)
  overlay = f"frame: {packet.frame_id}\nstate: {state}\nop: {operation}\nfps: {fps:.2f}"
  vis = cv_draw_text(vis, text=overlay, uv_top_left=(10, 10), color=(255, 255, 0), fontScale=0.6, thickness=2, outline_color=(0, 0, 0))
  return vis


def print_pose_report(packet: FramePacket, pose, state, operation, processing_time, fps):
  timestamp = "None" if packet.timestamp is None else f"{packet.timestamp:.9f}"
  print(f"frame_id: {packet.frame_id}")
  print(f"timestamp_metadata: {timestamp}")
  print(f"state: {state}")
  print(f"operation: {operation}")
  print(f"processing_time_sec: {processing_time:.6f}")
  print(f"processing_fps: {fps:.3f}")
  print("pose:")
  print(pose.reshape(4, 4))
  print("")


def run_live(args):
  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  shutil.rmtree(f'{debug_dir}/ob_in_cam', ignore_errors=True)
  shutil.rmtree(f'{debug_dir}/track_vis', ignore_errors=True)
  os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
  if args.save_track_vis:
    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  if args.provider == "recorded":
    provider = RecordedFoundationPoseProvider(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  elif args.provider == "network":
    provider = NetworkFrameProvider(host=args.host, port=args.port)
  else:
    raise RuntimeError(f"unknown provider: {args.provider}")

  try:
    state = "UNREGISTERED"
    while True:
      packet = provider.get_frame()
      if packet is None:
        logging.info("frame provider reached end of stream")
        break

      validate_frame_packet(packet)
      logging.info(f'i:{packet.index}')

      start_time = time.perf_counter()
      if state == "UNREGISTERED":
        if packet.mask is None:
          raise RuntimeError("first registration frame requires a mask")
        operation = "REGISTER"
        pose = est.register(K=packet.K, rgb=packet.rgb, depth=packet.depth, ob_mask=packet.mask, iteration=args.est_refine_iter)
        state = "TRACKING"
      elif state == "TRACKING":
        operation = "TRACK"
        pose = est.track_one(rgb=packet.rgb, depth=packet.depth, K=packet.K, iteration=args.track_refine_iter)
      else:
        raise RuntimeError(f"unknown state: {state}")

      processing_time = time.perf_counter() - start_time
      fps = 1.0 / processing_time if processing_time > 0 else np.inf

      np.savetxt(f'{debug_dir}/ob_in_cam/{packet.frame_id}.txt', pose.reshape(4,4))
      print_pose_report(packet=packet, pose=pose, state=state, operation=operation, processing_time=processing_time, fps=fps)

      if debug >= 1 or args.save_track_vis:
        vis = draw_live_overlay(packet=packet, pose=pose, to_origin=to_origin, bbox=bbox, state=state, operation=operation, fps=fps)

      if debug >= 1:
        cv2.imshow('FoundationPose Live', vis[...,::-1])
        key = cv2.waitKey(1)
        if key == ord('q'):
          break

      if args.save_track_vis:
        imageio.imwrite(f'{debug_dir}/track_vis/{packet.frame_id}.png', vis)
  finally:
    if hasattr(provider, "close"):
      provider.close()
    if debug >= 1:
      cv2.destroyAllWindows()


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_live')
  parser.add_argument('--save_track_vis', action='store_true')
  parser.add_argument('--provider', choices=['recorded', 'network'], default='recorded')
  parser.add_argument('--host', type=str, default='0.0.0.0')
  parser.add_argument('--port', type=int, default=5000)
  args = parser.parse_args()
  run_live(args)
