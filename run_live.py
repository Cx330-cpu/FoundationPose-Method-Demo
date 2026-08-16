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
import glob
import json
import math
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time
import shutil
import socket
import cv2
import imageio.v2 as imageio
import numpy as np
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
    self.video_dir = video_dir
    self.zfar = zfar
    self.model_free_layout = os.path.isfile(f'{video_dir}/K.txt')
    if self.model_free_layout:
      self.K = np.loadtxt(f'{video_dir}/K.txt').reshape(3, 3)
      self.color_files = sorted(glob.glob(f'{video_dir}/rgb/*.png'))
      if len(self.color_files) == 0:
        raise RuntimeError(f"no recorded RGB frames found in {video_dir}/rgb")
      self.id_strs = [os.path.basename(p).replace('.png', '') for p in self.color_files]
    else:
      self.reader = YcbineoatReader(video_dir=video_dir, shorter_side=shorter_side, zfar=zfar)
    self.index = 0

  def __len__(self):
    if self.model_free_layout:
      return len(self.color_files)
    return len(self.reader.color_files)

  def _timestamp_from_frame_id(self, frame_id):
    try:
      return int(frame_id) * 1e-9
    except ValueError:
      return None

  def get_frame(self) -> Optional[FramePacket]:
    if self.index >= len(self):
      return None

    i = self.index
    if self.model_free_layout:
      frame_id = self.id_strs[i]
      color_file = self.color_files[i]
      rgb = imageio.imread(color_file)[..., :3]
      depth_file = color_file.replace('/rgb/', '/depth_enhanced/')
      mask_file = color_file.replace('/rgb/', '/mask/')
      depth_mm = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
      if depth_mm is None or depth_mm.dtype != np.uint16:
        raise RuntimeError(f"depth frame must be uint16 mm: {depth_file}")
      depth = depth_mm.astype(np.float32) / 1000.0
      depth[(depth < 0.001) | (depth >= self.zfar)] = 0
      if i == 0:
        mask_img = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
          raise RuntimeError(f"first registration frame requires mask: {mask_file}")
        mask = mask_img > 0
      else:
        mask = None
      K = self.K
    else:
      frame_id = self.reader.id_strs[i]
      rgb = self.reader.get_color(i)
      depth = self.reader.get_depth(i)
      mask = self.reader.get_mask(0).astype(bool) if i == 0 else None
      K = self.reader.K
    packet = FramePacket(
      rgb=rgb,
      depth=depth,
      K=K,
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

  def send_control(self, payload):
    payload = {
      "magic": "FPCONTROL",
      "version": 1,
      **payload,
    }
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    self.client_sock.sendall(struct.pack(">I", len(data)) + data)

  def request_mask(self, frame_id, reason, request_frames):
    self.send_control({
      "type": "mask_request",
      "frame_id": str(frame_id),
      "reason": str(reason),
      "request_frames": int(request_frames),
    })

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


def bbox_corners(bbox):
  lo, hi = bbox
  return np.array(
    [[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]],
    dtype=np.float64,
  )


def rotation_geodesic_angle(R1, R2):
  rel = R1.T @ R2
  value = (np.trace(rel) - 1.0) * 0.5
  return float(math.acos(float(np.clip(value, -1.0, 1.0))))


def projected_bbox_mask_overlap(K, pose, to_origin, bbox, mask):
  if mask is None or mask.sum() == 0:
    return None
  center_pose = pose @ np.linalg.inv(to_origin)
  pts = bbox_corners(bbox)
  pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
  pts_cam = (center_pose @ pts_h.T).T[:, :3]
  valid = pts_cam[:, 2] > 0.001
  if valid.sum() < 2:
    return 0.0
  pts_cam = pts_cam[valid]
  u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
  v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
  H, W = mask.shape
  x0, x1 = int(np.floor(np.min(u))), int(np.ceil(np.max(u)))
  y0, y1 = int(np.floor(np.min(v))), int(np.ceil(np.max(v)))
  x0, x1 = max(0, x0), min(W - 1, x1)
  y0, y1 = max(0, y0), min(H - 1, y1)
  if x1 <= x0 or y1 <= y0:
    return 0.0
  roi = mask[y0 : y1 + 1, x0 : x1 + 1] > 0
  return float(roi.mean())


def projected_bbox_rect(K, pose, to_origin, bbox, image_shape):
  center_pose = pose @ np.linalg.inv(to_origin)
  pts = bbox_corners(bbox)
  pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
  pts_cam = (center_pose @ pts_h.T).T[:, :3]
  valid = pts_cam[:, 2] > 0.001
  if valid.sum() < 2:
    return None
  pts_cam = pts_cam[valid]
  u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
  v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
  H, W = image_shape
  raw_x0, raw_x1 = float(np.min(u)), float(np.max(u))
  raw_y0, raw_y1 = float(np.min(v)), float(np.max(v))
  x0, x1 = max(0, int(np.floor(raw_x0))), min(W - 1, int(np.ceil(raw_x1)))
  y0, y1 = max(0, int(np.floor(raw_y0))), min(H - 1, int(np.ceil(raw_y1)))
  raw_area = max(1.0, (raw_x1 - raw_x0 + 1.0) * (raw_y1 - raw_y0 + 1.0))
  clipped_area = max(0.0, (x1 - x0 + 1.0) * (y1 - y0 + 1.0)) if x1 > x0 and y1 > y0 else 0.0
  return {
    "rect": (x0, y0, x1, y1),
    "visible_ratio": float(clipped_area / raw_area),
  }


def tracking_lost_reason(packet, pose, previous_pose, to_origin, bbox, args):
  rect_info = projected_bbox_rect(packet.K, pose, to_origin, bbox, packet.depth.shape)
  if rect_info is None:
    return "bbox_not_projectable"
  if rect_info["visible_ratio"] < args.lost_min_bbox_visible_ratio:
    return f"bbox_out_of_view={rect_info['visible_ratio']:.3f}"
  x0, y0, x1, y1 = rect_info["rect"]
  roi = packet.depth[y0 : y1 + 1, x0 : x1 + 1]
  if roi.size == 0:
    return "empty_projected_bbox"
  depth_ratio = float((roi > 0.001).mean())
  if depth_ratio < args.lost_min_bbox_depth_ratio:
    return f"low_projected_depth={depth_ratio:.3f}"
  if previous_pose is not None:
    trans_delta = float(np.linalg.norm(pose[:3, 3] - previous_pose[:3, 3]))
    rot_delta = math.degrees(rotation_geodesic_angle(previous_pose[:3, :3], pose[:3, :3]))
    if trans_delta > args.lost_max_translation_delta:
      return f"pose_translation_jump={trans_delta:.3f}"
    if rot_delta > args.lost_max_rotation_delta_deg:
      return f"pose_rotation_jump={rot_delta:.1f}"
  return None


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
    last_register_index = None
    last_mask_request_index = None
    previous_pose = None
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
        last_register_index = packet.index
        state = "TRACKING"
      elif state == "TRACKING":
        operation = "TRACK"
        pose = est.track_one(rgb=packet.rgb, depth=packet.depth, K=packet.K, iteration=args.track_refine_iter)
        if args.mask_recovery and packet.mask is not None:
          frames_since_register = packet.index - last_register_index if last_register_index is not None else np.inf
          overlap = projected_bbox_mask_overlap(packet.K, pose, to_origin, bbox, packet.mask)
          can_recover = frames_since_register >= args.mask_recovery_min_gap
          periodic = args.mask_recovery_interval > 0 and frames_since_register >= args.mask_recovery_interval
          low_overlap = overlap is not None and overlap < args.mask_recovery_min_overlap
          if can_recover and (periodic or low_overlap):
            reason = "interval" if periodic else f"mask_overlap={overlap:.3f}"
            operation = f"RE-REGISTER({reason})"
            logging.info(f"mask recovery triggered on frame {packet.frame_id}: {reason}")
            pose = est.register(K=packet.K, rgb=packet.rgb, depth=packet.depth, ob_mask=packet.mask, iteration=args.est_refine_iter)
            last_register_index = packet.index
        elif args.mask_request and packet.mask is None and hasattr(provider, "request_mask"):
          reason = tracking_lost_reason(packet, pose, previous_pose, to_origin, bbox, args)
          cooldown_ok = last_mask_request_index is None or packet.index - last_mask_request_index >= args.mask_request_cooldown
          if reason is not None and cooldown_ok:
            provider.request_mask(packet.frame_id, reason, args.mask_request_frames)
            last_mask_request_index = packet.index
            logging.info(f"sent mask_request frame={packet.frame_id} reason={reason} request_frames={args.mask_request_frames}")
      else:
        raise RuntimeError(f"unknown state: {state}")

      processing_time = time.perf_counter() - start_time
      fps = 1.0 / processing_time if processing_time > 0 else np.inf

      np.savetxt(f'{debug_dir}/ob_in_cam/{packet.frame_id}.txt', pose.reshape(4,4))
      print_pose_report(packet=packet, pose=pose, state=state, operation=operation, processing_time=processing_time, fps=fps)
      previous_pose = pose.copy()

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
  parser.add_argument('--mask_recovery', action='store_true', default=True)
  parser.add_argument('--no_mask_recovery', dest='mask_recovery', action='store_false')
  parser.add_argument('--mask_recovery_interval', type=int, default=8)
  parser.add_argument('--mask_recovery_min_gap', type=int, default=3)
  parser.add_argument('--mask_recovery_min_overlap', type=float, default=0.15)
  parser.add_argument('--mask_request', action='store_true', default=True)
  parser.add_argument('--no_mask_request', dest='mask_request', action='store_false')
  parser.add_argument('--mask_request_frames', type=int, default=5)
  parser.add_argument('--mask_request_cooldown', type=int, default=10)
  parser.add_argument('--lost_min_bbox_visible_ratio', type=float, default=0.35)
  parser.add_argument('--lost_min_bbox_depth_ratio', type=float, default=0.05)
  parser.add_argument('--lost_max_translation_delta', type=float, default=0.25)
  parser.add_argument('--lost_max_rotation_delta_deg', type=float, default=55.0)
  args = parser.parse_args()
  run_live(args)
