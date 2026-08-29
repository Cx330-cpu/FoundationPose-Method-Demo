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
import threading
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
  pc_received_time: float


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
      pc_received_time=time.time(),
    )
    self.index += 1
    return packet


class NetworkFrameProvider(FrameProvider):
  def __init__(self, host, port, latest_only=False):
    self.host = host
    self.port = port
    self.latest_only = latest_only
    self.latest_packet = None
    self.stream_ended = False
    self.lock = threading.Lock()
    self.send_lock = threading.Lock()
    self.new_frame_event = threading.Event()
    self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.server_sock.bind((host, port))
    self.server_sock.listen(1)
    logging.info(f"waiting for frame sender on {host}:{port}")
    self.client_sock, self.client_addr = self.server_sock.accept()
    logging.info(f"accepted frame sender from {self.client_addr}")
    self.n_received = 0
    self.n_dropped = 0
    self.n_emitted = 0
    if self.latest_only:
      self.receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
      self.receiver_thread.start()

  def _receive_next_packet(self) -> Optional[FramePacket]:
    try:
      decoded = receive_frame(self.client_sock)
    except StreamEnd:
      return None

    header = decoded["header"]
    self.n_received += 1
    return FramePacket(
      rgb=decoded["rgb"],
      depth=decoded["depth"],
      K=decoded["K"],
      mask=decoded["mask"],
      timestamp=header["timestamp"],
      frame_id=header["frame_id"],
      index=header["index"],
      pc_received_time=time.time(),
    )

  def _receive_loop(self):
    while True:
      packet = self._receive_next_packet()
      with self.lock:
        if packet is None:
          self.stream_ended = True
          self.new_frame_event.set()
          return
        if self.latest_packet is not None:
          self.n_dropped += 1
        self.latest_packet = packet
        self.new_frame_event.set()

  def get_frame(self) -> Optional[FramePacket]:
    if not self.latest_only:
      packet = self._receive_next_packet()
      if packet is not None:
        self.n_emitted += 1
      return packet

    while True:
      self.new_frame_event.wait(timeout=0.1)
      with self.lock:
        if self.latest_packet is not None:
          packet = self.latest_packet
          self.latest_packet = None
          self.n_emitted += 1
          if not self.stream_ended:
            self.new_frame_event.clear()
          return packet
        if self.stream_ended:
          return None

  def send_json_message(self, magic, payload):
    payload = {
      "magic": magic,
      "version": 1,
      **payload,
    }
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with self.send_lock:
      self.client_sock.sendall(struct.pack(">I", len(data)) + data)

  def send_control(self, payload):
    self.send_json_message("FPCONTROL", payload)

  def request_mask(self, frame_id, reason, request_frames):
    self.send_control({
      "type": "mask_request",
      "frame_id": str(frame_id),
      "reason": str(reason),
      "request_frames": int(request_frames),
    })

  def send_result(self, payload):
    self.send_json_message("FPRESULT", payload)

  def close(self):
    logging.info(
      f"network frame stats received={getattr(self, 'n_received', 0)} "
      f"emitted={getattr(self, 'n_emitted', 0)} dropped={getattr(self, 'n_dropped', 0)} "
      f"latest_only={self.latest_only}"
    )
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


def rotation_matrix_to_quat(R):
  m = np.asarray(R, dtype=np.float64)
  trace = np.trace(m)
  if trace > 0:
    s = math.sqrt(trace + 1.0) * 2.0
    q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
  else:
    i = int(np.argmax(np.diag(m)))
    if i == 0:
      s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
      q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif i == 1:
      s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
      q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
      s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
      q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
  return q / max(np.linalg.norm(q), 1e-12)


def quat_to_rotation_matrix(q):
  w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def slerp_quat(q0, q1, alpha):
  q0 = q0 / max(np.linalg.norm(q0), 1e-12)
  q1 = q1 / max(np.linalg.norm(q1), 1e-12)
  dot = float(np.dot(q0, q1))
  if dot < 0.0:
    q1 = -q1
    dot = -dot
  if dot > 0.9995:
    q = q0 + alpha * (q1 - q0)
    return q / max(np.linalg.norm(q), 1e-12)
  theta_0 = math.acos(float(np.clip(dot, -1.0, 1.0)))
  theta = theta_0 * alpha
  sin_theta = math.sin(theta)
  sin_theta_0 = math.sin(theta_0)
  s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
  s1 = sin_theta / sin_theta_0
  return s0 * q0 + s1 * q1


def smooth_pose(previous_pose, current_pose, alpha):
  if previous_pose is None or alpha >= 1.0:
    return current_pose.copy()
  alpha = float(np.clip(alpha, 0.0, 1.0))
  out = current_pose.copy()
  out[:3, 3] = (1.0 - alpha) * previous_pose[:3, 3] + alpha * current_pose[:3, 3]
  q0 = rotation_matrix_to_quat(previous_pose[:3, :3])
  q1 = rotation_matrix_to_quat(current_pose[:3, :3])
  out[:3, :3] = quat_to_rotation_matrix(slerp_quat(q0, q1, alpha))
  return out


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


def project_points_2d(K, pose, pts):
  pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
  pts_cam = (pose @ pts_h.T).T[:, :3]
  valid = pts_cam[:, 2] > 0.001
  uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
  if valid.any():
    uv[valid, 0] = K[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2] + K[0, 2]
    uv[valid, 1] = K[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2] + K[1, 2]
  return uv, valid


def line_from_uv(uv, valid, i, j):
  if not (valid[i] and valid[j]):
    return None
  return [[float(uv[i, 0]), float(uv[i, 1])], [float(uv[j, 0]), float(uv[j, 1])]]


def make_pose_result(packet, pose, to_origin, bbox, state, operation, fps):
  center_pose = pose @ np.linalg.inv(to_origin)
  corners = bbox_corners(bbox)
  uv, valid = project_points_2d(packet.K, center_pose, corners)
  # bbox_corners order is x outer, y middle, z inner. Edges connect points
  # whose corner coordinates differ along exactly one axis.
  bbox_lines = []
  for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
      if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
        line = line_from_uv(uv, valid, i, j)
        if line is not None:
          bbox_lines.append(line)

  scale = 0.1
  axis_pts = np.array(
    [
      [0.0, 0.0, 0.0],
      [scale, 0.0, 0.0],
      [0.0, scale, 0.0],
      [0.0, 0.0, scale],
    ],
    dtype=np.float64,
  )
  axis_uv, axis_valid = project_points_2d(packet.K, center_pose, axis_pts)
  axis_specs = [
    ("x", [255, 0, 0], 1),
    ("y", [0, 255, 0], 2),
    ("z", [0, 0, 255], 3),
  ]
  axis_lines = []
  for name, color, idx in axis_specs:
    line = line_from_uv(axis_uv, axis_valid, 0, idx)
    if line is not None:
      axis_lines.append({"name": name, "color": color, "from": line[0], "to": line[1]})

  H, W = packet.rgb.shape[:2]
  return {
    "type": "tracking_result",
    "frame_id": str(packet.frame_id),
    "index": int(packet.index),
    "timestamp": packet.timestamp,
    "state": str(state),
    "operation": str(operation),
    "processing_fps": float(fps),
    "image_width": int(W),
    "image_height": int(H),
    "ob_in_cam": pose.reshape(4, 4).astype(float).tolist(),
    "bbox_lines_2d": bbox_lines,
    "axis_lines_2d": axis_lines,
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


def print_pose_report(packet: FramePacket, pose, state, operation, processing_time, fps, queue_latency_ms=None):
  timestamp = "None" if packet.timestamp is None else f"{packet.timestamp:.9f}"
  print(f"frame_id: {packet.frame_id}")
  print(f"timestamp_metadata: {timestamp}")
  print(f"state: {state}")
  print(f"operation: {operation}")
  print(f"processing_time_sec: {processing_time:.6f}")
  print(f"processing_fps: {fps:.3f}")
  if queue_latency_ms is not None:
    print(f"pc_queue_latency_ms: {queue_latency_ms:.3f}")
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
    provider = NetworkFrameProvider(host=args.host, port=args.port, latest_only=args.latest_frame_only)
  else:
    raise RuntimeError(f"unknown provider: {args.provider}")

  try:
    state = "UNREGISTERED"
    last_register_index = None
    last_mask_request_index = None
    previous_pose = None
    previous_display_pose = None
    while True:
      packet = provider.get_frame()
      if packet is None:
        logging.info("frame provider reached end of stream")
        break

      validate_frame_packet(packet)
      logging.info(f'i:{packet.index}')

      wall_start_time = time.time()
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
      display_pose = smooth_pose(previous_display_pose, pose, args.result_smoothing_alpha)
      previous_display_pose = display_pose.copy()
      queue_latency_ms = float(max(0.0, wall_start_time - packet.pc_received_time) * 1000.0)

      if args.send_results and hasattr(provider, "send_result"):
        try:
          result = make_pose_result(packet, display_pose, to_origin, bbox, state, operation, fps)
          result["raw_ob_in_cam"] = pose.reshape(4, 4).astype(float).tolist()
          result["processing_time_sec"] = float(processing_time)
          result["pc_received_time"] = float(packet.pc_received_time)
          result["pc_result_time"] = float(time.time())
          result["pc_queue_latency_ms"] = queue_latency_ms
          result["smoothing_alpha"] = float(args.result_smoothing_alpha)
          provider.send_result(result)
        except Exception as exc:
          logging.info(f"failed to send FPRESULT for frame {packet.frame_id}: {exc}")

      np.savetxt(f'{debug_dir}/ob_in_cam/{packet.frame_id}.txt', pose.reshape(4,4))
      print_pose_report(packet=packet, pose=pose, state=state, operation=operation, processing_time=processing_time, fps=fps, queue_latency_ms=queue_latency_ms)
      previous_pose = pose.copy()

      if debug >= 1 or args.save_track_vis:
        vis = draw_live_overlay(packet=packet, pose=display_pose, to_origin=to_origin, bbox=bbox, state=state, operation=operation, fps=fps)

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
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/polycam_verify/ob_0000001/model/model.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/polycam_verify/ob_0000001')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_live')
  parser.add_argument('--save_track_vis', action='store_true')
  parser.add_argument('--provider', choices=['recorded', 'network'], default='recorded')
  parser.add_argument('--host', type=str, default='0.0.0.0')
  parser.add_argument('--port', type=int, default=5000)
  parser.add_argument('--mask_recovery', action='store_true', default=False)
  parser.add_argument('--no_mask_recovery', dest='mask_recovery', action='store_false')
  parser.add_argument('--mask_recovery_interval', type=int, default=0)
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
  parser.add_argument('--send_results', action='store_true', default=True)
  parser.add_argument('--no_send_results', dest='send_results', action='store_false')
  parser.add_argument('--latest_frame_only', action='store_true', default=True)
  parser.add_argument('--no_latest_frame_only', dest='latest_frame_only', action='store_false')
  parser.add_argument('--result_smoothing_alpha', type=float, default=1.0)
  args = parser.parse_args()
  run_live(args)
