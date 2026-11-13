"""Fast SAM 3D Body multi-view SMPL publisher for IGRIS video teleop (runs on the GPU PC).

Wraps upstream `run_multiview_publisher.py` (unchanged) with this machine's asset paths
and RealSense modes that the cameras actually support. Run in the `fast_sam_3d_body` env:

    python run_fastsam_multiview.py --list                       # connected RealSense cameras
    python run_fastsam_multiview.py --serials SN_MAIN,SN_SIDE    # 2 cameras, first = main (body yaw)
    python run_fastsam_multiview.py --serials SN_MAIN,SN_SIDE --record
    python run_fastsam_multiview.py --source video --videos a.mp4 b.mp4 --intrinsics a.json b.json

Publishes SMPL (SONIC protocol v2) on tcp://*:5556; the robot PC subscribes to it.
Paths can be changed with env vars FASTSAM_ROOT, SMPL_MODEL, MHR_ASSETS.
"""

import argparse
import os
import sys

HOME = os.path.expanduser("~")
FASTSAM = os.environ.get("FASTSAM_ROOT", f"{HOME}/third_party/Fast-SAM-3D-Body")
SMPL_MODEL = os.environ.get("SMPL_MODEL", f"{HOME}/IsaacLab/scripts/fastsam_sonic/assets/smpl/SMPL_NEUTRAL.pkl")
MHR_ASSETS = os.environ.get("MHR_ASSETS", f"{HOME}/third_party/mhr/tools/mhr_smpl_conversion/assets")

# Color modes per model (the upstream default 848x480 is not a D555 mode).
MODE_BY_MODEL = {"D555": (896, 504), "D455": (848, 480), "D435": (848, 480), "D415": (848, 480)}


def list_cameras():
    import pyrealsense2 as rs
    devs = list(rs.context().query_devices())
    if not devs:
        print("no RealSense camera detected")
        return
    for d in devs:
        name = d.get_info(rs.camera_info.name)
        sn = d.get_info(rs.camera_info.serial_number)
        sensors = [s.get_info(rs.camera_info.name) for s in d.query_sensors()]
        imu = any("Motion" in s for s in sensors)
        ci = getattr(rs.camera_info, "connection_type", None)
        conn = d.get_info(ci) if ci is not None and d.supports(ci) else "USB"
        conn = "PoE/Ethernet (DDS)" if conn == "DDS" else conn
        print(f"{sn}  {name}  {conn}  IMU={'yes' if imu else 'NO (gravity calibration needs an IMU: D435i/D455/D555)'}")


def auto_mode(serials):
    import pyrealsense2 as rs
    names = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
             for d in rs.context().query_devices()}
    modes = set()
    for sn in serials:
        name = names.get(sn, "")
        modes.add(next((m for k, m in MODE_BY_MODEL.items() if k in name), (848, 480)))
    if len(modes) > 1:
        raise SystemExit(f"cameras need different color modes {modes}; pass --width/--height explicitly")
    return modes.pop()


def make_camera_source(up, width, height, fps, serials, imu_samples=100):
    """Multi-RealSense source that also works for D555 over PoE (DDS).

    Upstream MultiRealSenseSource enables `rs.stream.accel`, which a D555 over Ethernet
    does not offer (only the combined `rs.stream.motion`), so start-up fails with
    "Couldn't resolve requests". Same interface as upstream; gravity is calibrated from
    whichever IMU stream the camera has, then the cameras stream color only.
    """
    import time

    import numpy as np
    import pyrealsense2 as rs

    class RealSenseSource(up.MultiCameraSource):
        def __init__(self):
            self.ctx = rs.context()
            devs = {d.get_info(rs.camera_info.serial_number): d for d in self.ctx.query_devices()}
            missing = [s for s in serials if s not in devs]
            if missing:
                raise RuntimeError(f"RealSense serials not found: {missing} (connected: {list(devs)})")
            self.serials, self.width, self.height, self._fps = list(serials), width, height, float(fps)
            self.cam_intrinsics, self.gravity_directions = [], []
            for sn in self.serials:
                imu = self._imu_stream(devs[sn])
                pipe = rs.pipeline(self.ctx)
                cfg = rs.config()
                cfg.enable_device(sn)
                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
                cfg.enable_stream(imu)
                prof = pipe.start(cfg)
                intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.cam_intrinsics.append(np.array(
                    [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)[None])
                acc, t_end = [], time.time() + 10.0
                while len(acc) < imu_samples and time.time() < t_end:
                    f = pipe.wait_for_frames(3000).first_or_default(imu)
                    if f:
                        m = f.as_motion_frame()
                        if imu == rs.stream.motion:
                            a = m.get_combined_motion_data().linear_acceleration
                        else:
                            a = m.get_motion_data()
                        acc.append([a.x, a.y, a.z])
                pipe.stop()
                if len(acc) < 10:
                    raise RuntimeError(f"no IMU samples from {sn}")
                g = -np.mean(acc, axis=0)
                g /= np.linalg.norm(g)
                self.gravity_directions.append(g)
                print(f"[igris-fastsam] {sn}: {devs[sn].get_info(rs.camera_info.name)} "
                      f"({self._conn(devs[sn])}) gravity [{g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}] "
                      f"from {len(acc)} {imu} samples", flush=True)
            self.pipelines = []
            for sn in self.serials:
                pipe = rs.pipeline(self.ctx)
                cfg = rs.config()
                cfg.enable_device(sn)
                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
                pipe.start(cfg)
                self.pipelines.append(pipe)

        @staticmethod
        def _imu_stream(dev):
            kinds = {p.stream_type() for s in dev.query_sensors() for p in s.get_stream_profiles()}
            if rs.stream.accel in kinds:
                return rs.stream.accel
            if rs.stream.motion in kinds:
                return rs.stream.motion
            raise RuntimeError("camera has no IMU (gravity calibration needs D435i/D455/D555)")

        @staticmethod
        def _conn(dev):
            ci = getattr(rs.camera_info, "connection_type", None)
            return dev.get_info(ci) if ci is not None and dev.supports(ci) else "USB"

        def get_frames(self):
            frames = []
            for pipe in self.pipelines:
                c = pipe.wait_for_frames(3000).get_color_frame()
                if not c:
                    raise RuntimeError("RealSense returned no color frame")
                frames.append(np.asanyarray(c.get_data()))
            return frames, time.time()

        def release(self):
            for pipe in self.pipelines:
                try:
                    pipe.stop()
                except Exception:
                    pass

        @property
        def fps(self):
            return self._fps

        def get_frame_size(self):
            return self.width, self.height

        def get_camera_intrinsics(self):
            return self.cam_intrinsics

        def get_gravity_directions(self):
            return self.gravity_directions

        def get_camera_names(self):
            return self.serials

    return RealSenseSource()


# --------------------------------------------------------------------------- target lock
# Upstream picks the largest person box in every camera on every frame, independently.
# With several people in view that can switch person between frames, or fuse two
# different people from the two cameras. Here each camera LOCKS one person instead:
#   acquire: the largest box, only if clearly larger than the next one (no lock otherwise)
#   track:   afterwards only the box that overlaps the previous one (IoU, similar size)
#   lost:    target not seen for TARGET_LOST_S -> unlock and acquire again (new target_id)
# The side camera re-locks whenever the main camera gets a new target and is stricter
# (a wrong side lock would fuse two people). target_id / target_ok go into the message;
# the robot side pauses when target_id changes or target_ok is 0.
TARGET_LOST_S = 0.7
ACQUIRE_RATIO = {"main": 1.5, "side": 2.5}   # largest area / second largest needed to lock
MIN_IOU = 0.15
TARGET = {"id": 0, "ok_t": 0.0}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _CamTarget:
    def __init__(self, name):
        self.name, self.box, self.seen, self.note = name, None, 0.0, ""

    def reset(self):
        self.box = None

    def select(self, boxes, now):
        import numpy as np
        boxes = np.zeros((0, 4), np.float32) if boxes is None else np.asarray(boxes, np.float32).reshape(-1, 4)
        if self.box is not None and now - self.seen > TARGET_LOST_S:
            self._say(f"target lost ({len(boxes)} people in view) - will lock again")
            self.box = None
        if self.box is not None:
            if len(boxes):
                ious = np.array([_iou(b, self.box) for b in boxes])
                k = int(np.argmax(ious))
                area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
                r = area(boxes[k]) / max(area(self.box), 1e-6)
                if ious[k] >= MIN_IOU and 0.5 < r < 2.0:
                    self.box, self.seen = boxes[k].copy(), now
                    return boxes[k:k + 1]
            return None                                   # target hidden this frame (others ignored)
        if not len(boxes):
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)
        if len(boxes) > 1 and areas[order[0]] < ACQUIRE_RATIO[self.name] * areas[order[1]]:
            self._say(f"{len(boxes)} people of similar size - not locking (only the operator should be close)")
            return None
        k = int(order[0])
        self.box, self.seen = boxes[k].copy(), now
        if self.name == "main":
            TARGET["id"] += 1
        self._say(f"locked target #{TARGET['id']} ({len(boxes)} people in view)")
        return boxes[k:k + 1]

    def _say(self, msg):
        if msg != self.note:
            self.note = msg
            print(f"[igris-fastsam] {self.name} camera: {msg}", flush=True)


def install_target_lock(up):
    import time
    cams = {}
    ctx = {"k": 0, "main": 0}
    _proc = up.RealtimeMultiViewPublisher._process_two_images_batched

    def pick_main_bbox(boxes):
        k = ctx["k"]                                      # upstream calls this for cam 0, then cam 1
        ctx["k"] += 1
        name = "main" if k == ctx["main"] else "side"
        t = cams.setdefault(k, _CamTarget(name))
        before = TARGET["id"]
        out = t.select(boxes, time.monotonic())
        if name == "main":
            if TARGET["id"] != before:                    # new main target -> side must re-lock too
                for j, o in cams.items():
                    if j != k:
                        o.reset()
            if out is not None:
                TARGET["ok_t"] = time.monotonic()
        return out

    def _process_two_images_batched(self, *a, **k):
        ctx["k"], ctx["main"] = 0, self.main_camera
        return _proc(self, *a, **k)

    up.pick_main_bbox = pick_main_bbox
    up.RealtimeMultiViewPublisher._process_two_images_batched = _process_two_images_batched


# MHR70 body keypoints (COCO-like order) and the sign that maps pred_keypoints_3d to
# OpenCV camera axes (x right, y down, z forward); checked on the test video.
KP_LSH, KP_RSH, KP_LHIP, KP_RHIP = 5, 6, 9, 10
HAND_EVERY = int(os.environ.get("HAND_EVERY", "3"))   # run the hand decoder every N body frames
import numpy as _np
KP_SIGN = _np.array([1.0, 1.0, 1.0])


def install_hand_closure(up):
    """Add a `hand_closure` (1, 2) [left, right] field to every published message.

    The upstream worker runs only the SAM 3D Body BODY decoder (fast, but its fingers are
    barely articulated: a fist reads as an open hand). Here, every HAND_EVERY frames, the
    two hand boxes the body decoder predicts on the MAIN camera are cropped and run through
    the HAND decoder (as SAM 3D Body's own full inference does; the left hand is mirrored
    so the model sees a right hand). Its finger keypoints give a 0..1 closure per hand
    (hand_closure.py), appended to the ZMQ message. Receivers that do not know the field
    (e.g. SONIC) can ignore it: the header lists it like any other field.
    """
    import threading
    import time

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hand_closure import hand_state

    latest = {"closure": np.full(2, np.nan), "r": np.full(2, np.nan), "t": 0.0}
    HANDS = {"n": 0, "ms": []}
    lock = threading.Lock()

    _build_est = up.build_default_estimator

    def build_default_estimator(*a, **k):
        est = _build_est(*a, **k)
        fwd = est.model.forward_step

        def forward_step(batch, *fa, **fk):
            out = fwd(batch, *fa, **fk)
            try:
                mhr = out.get("mhr") if isinstance(out, dict) else None
                kp = mhr.get("pred_keypoints_3d") if isinstance(mhr, dict) else None
                if kp is None:                       # warm-up / no person: nothing to keep
                    return out
                kp = kp.detach().float().cpu().numpy() if hasattr(kp, "detach") else np.asarray(kp)
                with lock:                           # body keypoints (heading) + inputs for the hand pass
                    latest.update(kp=kp, body_out=out, body_batch=batch)
            except Exception as e:  # never break the body pipeline because of the hands
                print(f"[igris-fastsam] keypoint capture failed: {e}", flush=True)
            return out

        HANDS.update(est=est, fwd=fwd)
        est.model.forward_step = forward_step
        return est

    up.build_default_estimator = build_default_estimator

    # Person position + heading (for walking), from the MAIN camera only, in a frame built
    # here from the camera and its IMU gravity (does not depend on upstream world conventions):
    #   e1 = camera optical axis projected on the floor (away from the camera), e3 = up,
    #   e2 = e3 x e1 (to the camera's left).  root_pos = [e1, e2, e3] m, root_heading = yaw of
    #   the person's facing direction in (e1, e2) [rad] (facing the camera = +-pi).
    root = {"p": np.full(3, np.nan), "h": np.nan, "t": 0.0, "dbg": ""}
    _proc = up.RealtimeMultiViewPublisher._process_two_images_batched

    def _process_two_images_batched(self, *a, **k):
        res = _proc(self, *a, **k)
        try:
            preds = res[0]
            m = preds[self.main_camera] if preds is not None else None
            if m is not None:
                row = sum(p is not None for p in preds[: self.main_camera])
                g = np.asarray(self.gravity_directions[self.main_camera], dtype=np.float64)  # down, OpenCV cam
                e3 = -g / np.linalg.norm(g)
                e1 = np.array([0.0, 0.0, 1.0]) - e3[2] * e3
                e1 /= np.linalg.norm(e1)
                E = np.stack([e1, np.cross(e3, e1), e3])            # rows: OpenCV cam -> floor frame
                cam_t = np.asarray(m["pred_cam_t"], dtype=np.float64).reshape(3)   # OpenCV, z = depth
                h, dbg = np.nan, ""
                with lock:
                    kp = latest.get("kp")
                if kp is not None and row < kp.shape[0]:
                    k3 = kp[row].astype(np.float64) * KP_SIGN               # -> OpenCV axes
                    left = (k3[KP_LHIP] - k3[KP_RHIP]) + (k3[KP_LSH] - k3[KP_RSH])
                    fwd = np.cross(left, e3)                                # body: x fwd = y_left x z_up
                    f = E @ fwd
                    if np.hypot(f[0], f[1]) > 1e-6:
                        h = float(np.arctan2(f[1], f[0]))
                    dbg = (f"head-hip y {k3[0][1] - (k3[KP_LHIP][1] + k3[KP_RHIP][1]) / 2:+.2f} "
                           f"nose-hip z {k3[0][2] - (k3[KP_LHIP][2] + k3[KP_RHIP][2]) / 2:+.2f} "
                           f"Lhip-Rhip x {k3[KP_LHIP][0] - k3[KP_RHIP][0]:+.2f}")
                with lock:
                    root.update(p=E @ cam_t, h=h, t=time.monotonic(), dbg=dbg)
        except Exception as e:
            print(f"[igris-fastsam] root position failed: {e}", flush=True)
        try:
            HANDS["n"] += 1
            preds = res[0]
            if preds is not None and preds[self.main_camera] is not None and HANDS["n"] % HAND_EVERY == 0:
                row = sum(p is not None for p in preds[: self.main_camera])
                _hand_pass(self, a[self.main_camera], row)
        except Exception as e:
            print(f"[igris-fastsam] hand decoder failed: {e}", flush=True)
        return res

    def _hand_pass(pub, img, row):
        import torch
        t0 = time.perf_counter()
        est, fwd, model = HANDS["est"], HANDS["fwd"], HANDS["est"].model
        with lock:
            out, batch = latest.get("body_out"), latest.get("body_batch")
        if out is None or "hand_box" not in out["mhr"]:
            return
        S = float(model.cfg.MODEL.IMAGE_SIZE[0])
        hb = out["mhr"]["hand_box"][row].detach().float().cpu().numpy() * S      # (2 [L, R], 4) crop px
        A = batch["affine_trans"].reshape(-1, 2, 3)[row].detach().float().cpu().numpy()   # full -> crop
        boxes = []
        for h in hb:                                    # same conversion as SAM3DBody._get_hand_box
            c = (h[:2] - A[:, 2]) / A[0, 0]
            half = h[2:].max() / A[0, 0] / 2
            boxes.append(np.array([[c[0] - half, c[1] - half, c[0] + half, c[1] + half]], np.float32))
        ci = batch["cam_int"]
        ci = ci.reshape(-1, 3, 3)[row:row + 1].clone()
        width = img.shape[1]
        lf = boxes[0].copy()                            # left hand: mirror image + box
        lf[:, 0], lf[:, 2] = width - boxes[0][:, 2] - 1, width - boxes[0][:, 0] - 1
        with torch.no_grad():
            bl = up.recursive_to(up.prepare_batch(np.ascontiguousarray(img[:, ::-1]), est.transform_hand, lf,
                                                  cam_int=ci.clone()), pub.device)
            br = up.recursive_to(up.prepare_batch(img, est.transform_hand, boxes[1], cam_int=ci.clone()), pub.device)
            bh = model._merge_hand_batches(bl, br)
            model._initialize_batch(bh)
            ho = fwd(bh, decoder_type="hand")
        key = "mhr_hand" if "mhr_hand" in ho else "mhr"
        kp = ho[key]["pred_keypoints_3d"].detach().float().cpu().numpy()       # (2, 70, 3): [mirrored L, R]
        # both crops are seen as RIGHT hands -> use the right-hand slot of each row
        cl, rl = hand_state(kp[0])
        cr, rr = hand_state(kp[1])
        c, r = np.array([cl[1], cr[1]]), np.array([rl[1], rr[1]])
        HANDS["ms"] = (HANDS["ms"] + [(time.perf_counter() - t0) * 1000])[-20:]
        HANDS["boxes"] = f"L box ctr {((boxes[0][0, :2] + boxes[0][0, 2:]) / 2).round().tolist()} R {((boxes[1][0, :2] + boxes[1][0, 2:]) / 2).round().tolist()} img {img.shape[1]}x{img.shape[0]}"
        with lock:
            latest.update(closure=c, r=r, t=time.monotonic())

    up.RealtimeMultiViewPublisher._process_two_images_batched = _process_two_images_batched

    class HandZMQPublisher(up.ZMQPublisher):
        def __init__(self, addr, *a, **k):
            super().__init__(addr, *a, **k)
            self.header_bytes = self._build_header([
                {"name": "frame_index", "dtype": "i64", "shape": [1]},
                {"name": "body_quat", "dtype": "f64", "shape": [1, 4]},
                {"name": "smpl_joints", "dtype": "f64", "shape": [1, 24, 3]},
                {"name": "smpl_pose", "dtype": "f64", "shape": [1, 21, 3]},
                {"name": "hand_closure", "dtype": "f64", "shape": [1, 2]},      # [left, right], 0 open .. 1 fist
                {"name": "hand_straightness", "dtype": "f64", "shape": [1, 2]}, # raw metric (for tuning)
                {"name": "root_pos", "dtype": "f64", "shape": [1, 3]},          # person [fwd-of-cam, left, up] m
                {"name": "root_heading", "dtype": "f64", "shape": [1, 1]},      # person facing yaw, rad
                {"name": "target_id", "dtype": "f64", "shape": [1, 1]},         # changes = different person
                {"name": "target_ok", "dtype": "f64", "shape": [1, 1]},         # 1 = locked person seen < 0.3 s ago
            ])
            self._last_print = 0.0

        def publish(self, body_quat, smpl_joints, smpl_pose):
            with lock:
                c, r, t = latest["closure"].copy(), latest["r"].copy(), latest["t"]
                p, tp, hd, dbg = root["p"].copy(), root["t"], root["h"], root["dbg"]
            if time.monotonic() - t > 0.8:      # no fresh hand estimate
                c[:] = np.nan
            if time.monotonic() - tp > 0.5:     # main camera lost the person
                p[:] = np.nan
                hd = np.nan
            data = (self.topic_bytes + self.header_bytes
                    + np.array([self.frame_idx], dtype=np.int64).tobytes()
                    + np.asarray(body_quat, dtype=np.float64).reshape(1, 4).tobytes()
                    + np.asarray(smpl_joints, dtype=np.float64).reshape(1, 24, 3).tobytes()
                    + np.asarray(smpl_pose, dtype=np.float64).reshape(1, 21, 3).tobytes()
                    + c.astype(np.float64).reshape(1, 2).tobytes()
                    + r.astype(np.float64).reshape(1, 2).tobytes()
                    + p.astype(np.float64).reshape(1, 3).tobytes()
                    + np.array([[hd]], dtype=np.float64).tobytes()
                    + np.array([[float(TARGET["id"])]], dtype=np.float64).tobytes()
                    + np.array([[1.0 if time.monotonic() - TARGET["ok_t"] < 0.3 else 0.0]], dtype=np.float64).tobytes())
            self.sock.send(data)
            self.frame_idx += 1
            if time.monotonic() - self._last_print > 2.0:
                self._last_print = time.monotonic()
                ms = np.mean(HANDS["ms"]) if HANDS["ms"] else float("nan")
                print(f"[igris-fastsam] hand decoder {ms:.0f} ms/{HAND_EVERY} frames | hands L/R closure {c[0]:.2f}/{c[1]:.2f} "
                      f"(straightness {r[0]:.2f}/{r[1]:.2f})  person at [{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}] m "
                      f"heading {np.degrees(hd):+.0f} deg  ({dbg}) {HANDS.get('boxes', '')}",
                      flush=True)

    up.ZMQPublisher = HandZMQPublisher


def main():
    ours = argparse.ArgumentParser(add_help=False)
    ours.add_argument("--list", action="store_true")
    ours.add_argument("--width", type=int, default=0)
    ours.add_argument("--height", type=int, default=0)
    ours.add_argument("--fps", type=int, default=30)
    o, rest = ours.parse_known_args()
    if o.list:
        list_cameras()
        return

    # upstream resolves checkpoints/ relative to its repo -> make user paths absolute before chdir
    rest = [os.path.abspath(a) if os.path.exists(a) else a for a in rest]
    sys.path.insert(0, FASTSAM)
    os.chdir(FASTSAM)
    defaults = {
        "--smpl-model-path": SMPL_MODEL,
        "--nn-model-dir": f"{FASTSAM}/mhr2smpl/experiments/multiview_n30000_e500",
        "--mhr2smpl-mapping-path": f"{MHR_ASSETS}/mhr2smpl_mapping.npz",
        "--mhr-mesh-path": f"{MHR_ASSETS}/mhr_face_mask.ply",
        "--smoother-dir": f"{FASTSAM}/mhr2smpl/experiments/smoother_w5",
    }
    engine = f"{FASTSAM}/checkpoints/yolo/yolo11m-pose.engine"
    defaults["--yolo-model"] = engine if os.path.exists(engine) else f"{FASTSAM}/checkpoints/yolo/yolo11m-pose.pt"
    argv = list(rest)
    for k, v in defaults.items():
        if k not in argv:
            argv += [k, v]
    sys.argv = [f"{FASTSAM}/run_multiview_publisher.py"] + argv
    print("[igris-fastsam]", " ".join(argv), flush=True)

    import run_multiview_publisher as up

    install_hand_closure(up)
    install_target_lock(up)          # after install_hand_closure (wraps its stage-1 hook)
    _orig = up.build_source

    def build_source(args):
        if args.source != "camera":
            return _orig(args)
        if not args.serials:
            raise SystemExit("--serials SN_MAIN,SN_SIDE is required (see --list)")
        serials = [s.strip() for s in args.serials.split(",") if s.strip()]
        w, h = (o.width, o.height) if o.width and o.height else auto_mode(serials)
        print(f"[igris-fastsam] cameras {serials} @ {w}x{h} {o.fps} fps (main = {serials[args.main_camera]})", flush=True)
        return make_camera_source(up, w, h, o.fps, serials)

    up.build_source = build_source
    up.main()


if __name__ == "__main__":
    main()
