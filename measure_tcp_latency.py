from datareader import *
import argparse
import csv
import json
import os
import socket
import struct
import threading
import time

from network_frame_protocol import ProtocolError, StreamEnd, recv_exact, send_frame


def timestamp_from_frame_id(frame_id):
  try:
    return int(frame_id) * 1e-9
  except ValueError:
    return None


def recv_json_message(sock):
  header_len_bytes = recv_exact(sock, 4, section="result_len", allow_clean_eof=True)
  header_len = struct.unpack(">I", header_len_bytes)[0]
  if header_len <= 0 or header_len > 8 * 1024 * 1024:
    raise ProtocolError(f"invalid JSON message length: {header_len}")
  payload = json.loads(recv_exact(sock, header_len, section="result_json").decode("utf-8"))
  recv_time = time.time()
  payload["_recv_time"] = recv_time
  return payload


class ResultReader:
  def __init__(self, sock):
    self.sock = sock
    self.lock = threading.Lock()
    self.events = {}
    self.messages = {}
    self.controls = []
    self.error = None
    self.closed = False
    self.thread = threading.Thread(target=self._loop, daemon=True)
    self.thread.start()

  def _loop(self):
    try:
      while True:
        try:
          msg = recv_json_message(self.sock)
        except StreamEnd:
          return
        magic = msg.get("magic")
        if magic == "FPRESULT":
          frame_id = str(msg.get("frame_id"))
          with self.lock:
            self.messages[frame_id] = msg
            event = self.events.get(frame_id)
            if event is not None:
              event.set()
        elif magic == "FPCONTROL":
          with self.lock:
            self.controls.append(msg)
          logging.info(f"received FPCONTROL type={msg.get('type')} frame={msg.get('frame_id')}")
        else:
          logging.info(f"ignored TCP JSON magic={magic}")
    except Exception as exc:
      self.error = exc
      logging.info(f"result reader stopped: {exc}")
    finally:
      self.closed = True
      with self.lock:
        for event in self.events.values():
          event.set()

  def wait_for(self, frame_id, timeout):
    frame_id = str(frame_id)
    with self.lock:
      if frame_id in self.messages:
        return self.messages[frame_id]
      if self.error is not None or self.closed:
        raise RuntimeError(f"result reader closed before {frame_id}: {self.error}")
      event = self.events.get(frame_id)
      if event is None:
        event = threading.Event()
        self.events[frame_id] = event
    if not event.wait(timeout=timeout):
      raise TimeoutError(f"timed out waiting for FPRESULT frame {frame_id}")
    with self.lock:
      if frame_id not in self.messages:
        raise RuntimeError(f"no FPRESULT for {frame_id}: {self.error}")
      return self.messages[frame_id]


def percentile(values, q):
  if values is None or len(values) == 0:
    return float("nan")
  arr = np.sort(np.asarray(values, dtype=float))
  if len(arr) == 1:
    return float(arr[0])
  idx = (len(arr) - 1) * (q / 100.0)
  lo = int(np.floor(idx))
  hi = int(np.ceil(idx))
  if lo == hi:
    return float(arr[lo])
  w = idx - lo
  return float(arr[lo] * (1.0 - w) + arr[hi] * w)


def summarize(name, values):
  if not values:
    print(f"{name}: n=0")
    return
  arr = np.asarray(values, dtype=float)
  print(
    f"{name}: n={len(arr)} median={np.median(arr):.3f} mean={np.mean(arr):.3f} "
    f"p95={percentile(arr, 95):.3f} min={arr.min():.3f} max={arr.max():.3f}"
  )


def write_csv(path, rows):
  if not rows:
    return
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def measure_tcp_latency(args):
  set_logging_format()
  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
  n_frames = len(reader.color_files) if args.max_frames <= 0 else min(args.max_frames, len(reader.color_files))
  frame_interval = 0 if args.fps <= 0 else 1.0 / args.fps
  ping_pong = args.mode == "pingpong"

  with socket.create_connection((args.host, args.port), timeout=args.connect_timeout) as sock:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    results = ResultReader(sock)
    logging.info(f"connected to {args.host}:{args.port} mode={args.mode} frames={n_frames} codec={args.rgb_codec}")
    rows = []
    send_meta = {}
    t_all0 = time.time()

    for i in range(n_frames):
      frame_start = time.perf_counter()
      frame_id = reader.id_strs[i]
      rgb = reader.get_color(i)
      depth = reader.get_depth(i)
      mask = reader.get_mask(0).astype(bool) if i == 0 else None
      t_send0 = time.time()
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
      t_send1 = time.time()
      send_meta[str(frame_id)] = (t_send0, t_send1, i)
      logging.info(f"sent frame {i}/{n_frames} id:{frame_id}")

      if ping_pong:
        timeout = args.register_timeout if i == 0 else args.result_timeout
        msg = results.wait_for(frame_id, timeout=timeout)
        rows.append(make_row(frame_id, i, t_send0, t_send1, msg))

      if frame_interval > 0:
        sleep_time = frame_interval - (time.perf_counter() - frame_start)
        if sleep_time > 0:
          time.sleep(sleep_time)

    if not ping_pong:
      try:
        sock.shutdown(socket.SHUT_WR)
      except OSError:
        pass
      deadline = time.time() + args.drain_timeout
      while time.time() < deadline:
        with results.lock:
          n_got = len(results.messages)
        if n_got >= n_frames or results.closed:
          break
        time.sleep(0.05)
      for frame_id, (t_send0, t_send1, index) in send_meta.items():
        with results.lock:
          msg = results.messages.get(frame_id)
        if msg is None:
          continue
        rows.append(make_row(frame_id, index, t_send0, t_send1, msg))

    wall_s = time.time() - t_all0
    write_csv(args.csv_path, rows)
    n_results = len(rows)
    n_dropped_client = n_frames - n_results
    print(f"mode: {args.mode}")
    print(f"codec: {args.rgb_codec}")
    print(f"fps_cap: {args.fps}")
    print(f"sent_frames: {n_frames}")
    print(f"fpresult_frames: {n_results}")
    print(f"client_unmatched_frames: {n_dropped_client}")
    print(f"wall_s: {wall_s:.6f}")
    print(f"send_wall_fps: {n_frames / wall_s if wall_s > 0 else 0:.3f}")
    print(f"result_wall_fps: {n_results / wall_s if wall_s > 0 else 0:.3f}")
    summarize("e2e_ms", [r["e2e_ms"] for r in rows])
    summarize("send_encode_tcp_ms", [r["send_encode_tcp_ms"] for r in rows])
    summarize("transfer_decode_ms", [r["transfer_decode_ms"] for r in rows])
    summarize("pc_queue_latency_ms", [r["pc_queue_latency_ms"] for r in rows])
    summarize("processing_ms", [r["processing_ms"] for r in rows])
    summarize("pc_to_result_ms", [r["pc_to_result_ms"] for r in rows])
    summarize("result_return_ms", [r["result_return_ms"] for r in rows])
    track_rows = [r for r in rows if r["operation"] == "TRACK"]
    summarize("e2e_ms_TRACK", [r["e2e_ms"] for r in track_rows])
    summarize("pc_queue_latency_ms_TRACK", [r["pc_queue_latency_ms"] for r in track_rows])
    print(f"csv_path: {args.csv_path}")


def make_row(frame_id, index, t_send0, t_send1, msg):
  t_recv = float(msg["_recv_time"])
  pc_received = float(msg.get("pc_received_time", t_send1))
  pc_result = float(msg.get("pc_result_time", t_recv))
  processing_sec = msg.get("processing_time_sec")
  if processing_sec is None:
    fps = msg.get("processing_fps")
    processing_sec = (1.0 / fps) if fps else 0.0
  return {
    "frame_id": str(frame_id),
    "index": int(index),
    "operation": str(msg.get("operation")),
    "state": str(msg.get("state")),
    "send_unix": t_send0,
    "e2e_ms": (t_recv - t_send0) * 1000.0,
    "send_encode_tcp_ms": (t_send1 - t_send0) * 1000.0,
    "transfer_decode_ms": max(0.0, (pc_received - t_send0) * 1000.0),
    "pc_queue_latency_ms": float(msg.get("pc_queue_latency_ms", 0.0)),
    "processing_ms": float(processing_sec) * 1000.0,
    "pc_to_result_ms": max(0.0, (pc_result - pc_received) * 1000.0),
    "result_return_ms": max(0.0, (t_recv - pc_result) * 1000.0),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=5000)
  parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/mustard0")
  parser.add_argument("--rgb_codec", choices=["png", "jpeg"], default="png")
  parser.add_argument("--jpeg_quality", type=int, default=95)
  parser.add_argument("--fps", type=float, default=0)
  parser.add_argument("--mode", choices=["pingpong", "paced"], default="pingpong")
  parser.add_argument("--max_frames", type=int, default=0)
  parser.add_argument("--connect_timeout", type=float, default=60)
  parser.add_argument("--register_timeout", type=float, default=30)
  parser.add_argument("--result_timeout", type=float, default=10)
  parser.add_argument("--drain_timeout", type=float, default=20)
  parser.add_argument("--csv_path", type=str, default=f"{code_dir}/results_today/tcp_latency/latency.csv")
  args = parser.parse_args()
  measure_tcp_latency(args)
