"""Publish SMPL poses in Fast SAM 3D Body's exact ZMQ format, without a camera or GPU.

    pixi run fake-stream                         # synthetic motion (arms, elbows, torso), 30 Hz
    pixi run fake-stream --npz smpl_xxx.npz      # replay a Fast-SAM recording (--record output)
Then point the teleop at it:  --addr tcp://localhost:5556
"""

import argparse
import json
import time

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from igris_teleop.smpl import forward_kinematics, pose_from_rotations

HEADER = json.dumps({"v": 2, "endian": "le", "count": 1, "fields": [
    {"name": "frame_index", "dtype": "i64", "shape": [1]},
    {"name": "body_quat", "dtype": "f64", "shape": [1, 4]},
    {"name": "smpl_joints", "dtype": "f64", "shape": [1, 24, 3]},
    {"name": "smpl_pose", "dtype": "f64", "shape": [1, 21, 3]},
    {"name": "hand_closure", "dtype": "f64", "shape": [1, 2]},
    {"name": "root_pos", "dtype": "f64", "shape": [1, 3]},
    {"name": "root_heading", "dtype": "f64", "shape": [1, 1]},
    {"name": "target_id", "dtype": "f64", "shape": [1, 1]},
    {"name": "target_ok", "dtype": "f64", "shape": [1, 1]},
]}, separators=(",", ":")).encode().ljust(1280, b"\0")


def R(axis, deg):
    return Rotation.from_euler(axis, deg, degrees=True).as_matrix()


def synthetic(t: float) -> np.ndarray:
    """Slow, robot-friendly motion: arms down -> forward/side reach, elbows bend, torso twist."""
    a = 0.5 - 0.5 * np.cos(2 * np.pi * t / 8.0)        # 0..1, 8 s period
    b = 0.5 - 0.5 * np.cos(2 * np.pi * t / 5.0)
    rots = {
        "left_shoulder": R("z", -80 + 30 * a) @ R("y", -50 * b),
        "right_shoulder": R("z", 80 - 50 * b) @ R("y", 40 * a),
        "left_elbow": R("y", -10 - 70 * a),
        "right_elbow": R("y", 10 + 80 * b),
        "spine2": R("y", 15 * np.sin(2 * np.pi * t / 10.0)),
        "spine1": R("x", 8 * a),
    }
    return pose_from_rotations(rots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="tcp://*:5556")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--npz", default=None, help="Fast-SAM recording (smpl_poses, body_quats, smpl_joints)")
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()

    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.addr)
    rec = np.load(args.npz) if args.npz else None
    print(f"[fake_stream] publishing on {args.addr} @ {args.hz:.0f} Hz "
          f"({'replay ' + args.npz if rec is not None else 'synthetic motion'}). Ctrl+C to stop.", flush=True)
    t0 = time.monotonic()
    i = 0
    try:
        while True:
            t = time.monotonic() - t0
            if args.duration and t > args.duration:
                break
            if rec is not None:
                k = i % len(rec["smpl_poses"])
                pose = rec["smpl_poses"][k].reshape(21, 3)
                quat = rec["body_quats"][k].reshape(4)
                joints = rec["smpl_joints"][k].reshape(24, 3)
            else:
                pose = synthetic(t)
                quat = np.array([1.0, 0, 0, 0])
                joints = forward_kinematics(pose)[0]
            # hands: left makes a fist for 3 s every 8 s, right for 2 s every 6 s (+ estimation noise)
            closure = np.array([0.9 if (t % 8.0) > 5.0 else 0.15, 0.9 if (t % 6.0) > 4.0 else 0.15])
            closure = np.clip(closure + 0.05 * np.random.standard_normal(2), 0, 1)
            # person 3 m in front of the camera, facing it (heading pi); every 16 s:
            # 4-8 s one step (0.4 m) toward the camera = forward, 10-13 s hips turned 35 deg left
            ph = t % 16.0
            fwd = 0.4 if 4.0 < ph < 8.0 else 0.0
            turn = np.radians(35.0) if 10.0 < ph < 13.0 else 0.0
            root = np.array([3.0 - fwd, 0.0, -0.8]) + 0.01 * np.random.standard_normal(3)
            heading = np.pi + turn + 0.02 * np.random.standard_normal()
            sock.send(b"pose" + HEADER + np.array([i], np.int64).tobytes() + quat.astype("<f8").tobytes()
                      + joints.astype("<f8").tobytes() + pose.astype("<f8").tobytes()
                      + (closure if rec is None else np.full(2, np.nan)).astype("<f8").tobytes()
                      + root.astype("<f8").tobytes() + np.array([heading], "<f8").tobytes()
                      # every 30 s: locked person hidden at 18-19.5 s, a different person locked from 24 s
                      + np.array([1.0 + (t // 30.0) * 2 + (1.0 if t % 30.0 > 24.0 else 0.0)], "<f8").tobytes()
                      + np.array([0.0 if 18.0 < t % 30.0 < 19.5 else 1.0], "<f8").tobytes())
            i += 1
            time.sleep(max(0.0, t0 + i / args.hz - time.monotonic()))
    except KeyboardInterrupt:
        pass
    sock.close(0)


if __name__ == "__main__":
    main()
