"""Receive SMPL poses from Fast SAM 3D Body (or SONIC-compatible publishers) over ZMQ.

Wire format (gear_sonic ZMQ protocol v2/v3, as sent by Fast-SAM `mocap/realtime/publisher.py`):
  b"pose" + 1280-byte JSON header (null padded) + raw little-endian arrays in header order
  header = {"v": 2, "endian": "le", "count": 1, "fields": [{"name", "dtype", "shape"}, ...]}
The header is parsed per message, so extra fields (v3) are accepted and kept.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np
import zmq

HEADER_SIZE = 1280
DTYPES = {"f64": "<f8", "f32": "<f4", "i64": "<i8", "i32": "<i4", "u8": "u1", "bool": "?"}


@dataclass
class SmplFrame:
    recv_time: float              # time.monotonic() at reception
    frame_index: int
    body_quat: np.ndarray         # (4,) WXYZ
    smpl_joints: np.ndarray       # (24, 3)
    smpl_pose: np.ndarray         # (21, 3)
    extra: dict


def parse_message(msg: bytes, topic: bytes = b"pose", header_size: int = HEADER_SIZE) -> dict[str, np.ndarray]:
    if not msg.startswith(topic):
        raise ValueError("unexpected topic")
    head = msg[len(topic):len(topic) + header_size].rstrip(b"\0")
    header = json.loads(head)
    off = len(topic) + header_size
    out = {}
    for f in header["fields"]:
        dt = np.dtype(DTYPES[f["dtype"]])
        n = int(np.prod(f["shape"]))
        out[f["name"]] = np.frombuffer(msg, dtype=dt, count=n, offset=off).reshape(f["shape"]).copy()
        off += n * dt.itemsize
    out["_version"] = header.get("v")
    return out


class SmplStream:
    """Background SUB socket keeping only the newest frame."""

    def __init__(self, addr: str, topic: str = "pose"):
        self.addr = addr
        self.topic = topic.encode()
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)      # newest message only
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        self._sock.connect(addr)
        self._lock = threading.Lock()
        self._latest: SmplFrame | None = None
        self._count = 0
        self._errors = 0
        self._times: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="smpl_stream", daemon=True)

    def start(self) -> "SmplStream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            now = time.monotonic()
            try:
                d = parse_message(msg, self.topic)
                fr = SmplFrame(
                    recv_time=now,
                    frame_index=int(np.asarray(d.get("frame_index", [-1])).reshape(-1)[0]),
                    body_quat=np.asarray(d["body_quat"], dtype=np.float64).reshape(4),
                    smpl_joints=np.asarray(d["smpl_joints"], dtype=np.float64).reshape(24, 3),
                    smpl_pose=np.asarray(d["smpl_pose"], dtype=np.float64).reshape(21, 3),
                    extra={k: v for k, v in d.items()
                           if k not in ("frame_index", "body_quat", "smpl_joints", "smpl_pose")},
                )
            except Exception:
                with self._lock:
                    self._errors += 1
                continue
            if not (np.isfinite(fr.smpl_pose).all() and np.isfinite(fr.smpl_joints).all()):
                with self._lock:
                    self._errors += 1
                continue
            with self._lock:
                self._latest = fr
                self._count += 1
                self._times.append(now)
                if len(self._times) > 200:
                    del self._times[:100]

    # ------------------------------------------------------------------ queries
    def latest(self) -> SmplFrame | None:
        with self._lock:
            return self._latest

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._latest is None else time.monotonic() - self._latest.recv_time

    def rate(self, window: float = 2.0) -> float:
        now = time.monotonic()
        with self._lock:
            ts = [t for t in self._times if now - t < window]
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 0.0

    def counts(self) -> tuple[int, int]:
        with self._lock:
            return self._count, self._errors
