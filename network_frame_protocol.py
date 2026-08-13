import json
import math
import struct

import cv2
import numpy as np


MAGIC = "FPFRAME"
VERSION = 1
MAX_HEADER_LEN = 1024 * 1024


class ProtocolError(RuntimeError):
  pass


class StreamEnd(EOFError):
  pass


def recv_exact(sock, n, section="unknown", frame_id=None, allow_clean_eof=False):
  chunks = []
  remaining = n
  while remaining:
    chunk = sock.recv(remaining)
    if not chunk:
      if allow_clean_eof and not chunks:
        raise StreamEnd("socket closed before next frame header")
      expected = n
      received = expected - remaining
      frame = "unknown" if frame_id is None else frame_id
      raise ProtocolError(f"EOF while reading {section} for frame {frame}: expected {expected} bytes, got {received}")
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


def _require(condition, message):
  if not condition:
    raise ProtocolError(message)


def _encode_image(name, image, ext, params=None):
  ok, encoded = cv2.imencode(ext, image, params or [])
  if not ok:
    raise ProtocolError(f"failed to encode {name} as {ext}")
  return encoded.tobytes()


def _decode_image(name, data, flags):
  arr = np.frombuffer(data, dtype=np.uint8)
  image = cv2.imdecode(arr, flags)
  if image is None:
    raise ProtocolError(f"failed to decode {name}")
  return image


def _timestamp_from_frame_id(frame_id):
  try:
    return int(frame_id) * 1e-9
  except (TypeError, ValueError):
    return None


def build_frame_message(rgb, depth, K, mask, frame_id, index, timestamp=None, rgb_codec="png", jpeg_quality=95):
  _require(rgb.ndim == 3 and rgb.shape[2] == 3, f"rgb must be (H, W, 3), got {rgb.shape}")
  _require(rgb.dtype == np.uint8, f"rgb must be uint8, got {rgb.dtype}")
  _require(depth.ndim == 2, f"depth must be (H, W), got {depth.shape}")
  _require(depth.shape == rgb.shape[:2], f"depth shape {depth.shape} does not match rgb shape {rgb.shape[:2]}")
  _require(np.asarray(K).shape == (3, 3), f"K must be (3, 3), got {np.asarray(K).shape}")
  if mask is not None:
    _require(mask.shape == depth.shape, f"mask shape {mask.shape} does not match depth shape {depth.shape}")

  rgb_codec = rgb_codec.lower()
  rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  if rgb_codec == "png":
    rgb_bytes = _encode_image("rgb", rgb_bgr, ".png")
  elif rgb_codec == "jpeg":
    rgb_bytes = _encode_image("rgb", rgb_bgr, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
  else:
    raise ProtocolError(f"unsupported rgb codec: {rgb_codec}")

  depth_mm = np.rint(depth * 1000.0).astype(np.uint16)
  depth_bytes = _encode_image("depth", depth_mm, ".png")

  if mask is None:
    mask_bytes = b""
    mask_format = "none"
  else:
    mask_u8 = (mask.astype(bool).astype(np.uint8) * 255)
    mask_bytes = _encode_image("mask", mask_u8, ".png")
    mask_format = "uint8_png"

  K = np.asarray(K)
  H, W = rgb.shape[:2]
  header = {
    "magic": MAGIC,
    "version": VERSION,
    "frame_id": str(frame_id),
    "index": int(index),
    "timestamp": _timestamp_from_frame_id(frame_id) if timestamp is None else timestamp,
    "width": int(W),
    "height": int(H),
    "fx": float(K[0, 0]),
    "fy": float(K[1, 1]),
    "cx": float(K[0, 2]),
    "cy": float(K[1, 2]),
    "rgb_format": rgb_codec,
    "depth_format": "uint16_png_mm",
    "mask_format": mask_format,
    "rgb_len": len(rgb_bytes),
    "depth_len": len(depth_bytes),
    "mask_len": len(mask_bytes),
  }
  header_bytes = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
  _require(0 < len(header_bytes) <= MAX_HEADER_LEN, f"header length out of range: {len(header_bytes)}")
  return struct.pack(">I", len(header_bytes)) + header_bytes + rgb_bytes + depth_bytes + mask_bytes


def send_frame(sock, rgb, depth, K, mask, frame_id, index, timestamp=None, rgb_codec="png", jpeg_quality=95):
  message = build_frame_message(
    rgb=rgb,
    depth=depth,
    K=K,
    mask=mask,
    frame_id=frame_id,
    index=index,
    timestamp=timestamp,
    rgb_codec=rgb_codec,
    jpeg_quality=jpeg_quality,
  )
  sock.sendall(message)


def _validate_header(header):
  for key in [
    "magic", "version", "frame_id", "index", "timestamp", "width", "height",
    "fx", "fy", "cx", "cy", "rgb_format", "depth_format", "mask_format",
    "rgb_len", "depth_len", "mask_len",
  ]:
    _require(key in header, f"missing header field: {key}")

  _require(header["magic"] == MAGIC, f"invalid magic: {header['magic']}")
  _require(header["version"] == VERSION, f"unsupported version: {header['version']}")
  _require(isinstance(header["width"], int) and header["width"] > 0, f"invalid width: {header['width']}")
  _require(isinstance(header["height"], int) and header["height"] > 0, f"invalid height: {header['height']}")
  _require(isinstance(header["index"], int) and header["index"] >= 0, f"invalid index: {header['index']}")
  _require(math.isfinite(header["fx"]) and header["fx"] > 0, f"invalid fx: {header['fx']}")
  _require(math.isfinite(header["fy"]) and header["fy"] > 0, f"invalid fy: {header['fy']}")
  _require(math.isfinite(header["cx"]), f"invalid cx: {header['cx']}")
  _require(math.isfinite(header["cy"]), f"invalid cy: {header['cy']}")
  if header["timestamp"] is not None:
    _require(math.isfinite(header["timestamp"]), f"invalid timestamp: {header['timestamp']}")
  _require(isinstance(header["rgb_len"], int) and header["rgb_len"] > 0, f"invalid rgb_len: {header['rgb_len']}")
  _require(isinstance(header["depth_len"], int) and header["depth_len"] > 0, f"invalid depth_len: {header['depth_len']}")
  _require(isinstance(header["mask_len"], int) and header["mask_len"] >= 0, f"invalid mask_len: {header['mask_len']}")
  _require(header["rgb_format"] in ["png", "jpeg"], f"unsupported rgb_format: {header['rgb_format']}")
  _require(header["depth_format"] == "uint16_png_mm", f"unsupported depth_format: {header['depth_format']}")
  if header["mask_len"] == 0:
    _require(header["mask_format"] == "none", f"mask_len is 0 but mask_format is {header['mask_format']}")
  else:
    _require(header["mask_format"] == "uint8_png", f"unsupported mask_format: {header['mask_format']}")


def receive_frame(sock):
  header_len_bytes = recv_exact(sock, 4, section="header_len", allow_clean_eof=True)
  header_len = struct.unpack(">I", header_len_bytes)[0]
  _require(0 < header_len <= MAX_HEADER_LEN, f"header length out of range: {header_len}")

  header = json.loads(recv_exact(sock, header_len, section="header").decode("utf-8"))
  _validate_header(header)
  frame_id = header["frame_id"]

  rgb_bytes = recv_exact(sock, header["rgb_len"], section="rgb", frame_id=frame_id)
  depth_bytes = recv_exact(sock, header["depth_len"], section="depth", frame_id=frame_id)
  mask_bytes = recv_exact(sock, header["mask_len"], section="mask", frame_id=frame_id) if header["mask_len"] else b""

  rgb_bgr = _decode_image("rgb", rgb_bytes, cv2.IMREAD_COLOR)
  rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
  depth_mm = _decode_image("depth", depth_bytes, cv2.IMREAD_UNCHANGED)
  _require(depth_mm.ndim == 2, f"decoded depth must be 2D, got {depth_mm.shape}")
  _require(depth_mm.dtype == np.uint16, f"decoded depth must be uint16, got {depth_mm.dtype}")
  depth = depth_mm / 1e3

  mask = None
  if mask_bytes:
    mask_img = _decode_image("mask", mask_bytes, cv2.IMREAD_UNCHANGED)
    _require(mask_img.ndim == 2, f"decoded mask must be 2D, got {mask_img.shape}")
    mask = mask_img > 0

  height = header["height"]
  width = header["width"]
  _require(rgb.shape == (height, width, 3), f"decoded rgb shape {rgb.shape} does not match header {(height, width, 3)}")
  _require(depth.shape == (height, width), f"decoded depth shape {depth.shape} does not match header {(height, width)}")
  if mask is not None:
    _require(mask.shape == (height, width), f"decoded mask shape {mask.shape} does not match header {(height, width)}")

  K = np.array([
    [header["fx"], 0.0, header["cx"]],
    [0.0, header["fy"], header["cy"]],
    [0.0, 0.0, 1.0],
  ])

  return {
    "header": header,
    "rgb": rgb,
    "depth": depth,
    "K": K,
    "mask": mask,
  }
