"""Video teleop main loop: SMPL stream -> retarget -> filter -> commander.

States
  HOLD    robot holds `hold_pose` (initial pose at start, the last command after a pause)
  ENGAGE  blends from hold_pose to the person's pose over control.engage_ramp_s
  TRACK   follows the person
Keys (terminal): ENTER engage / pause, q quit,
  g switch gripper mode (hand: fist = close / open hand = open, key: z / x toggle left / right).
Ctrl+C = q. Quitting returns to the initial pose first; a second Ctrl+C stops at once.
"""

from __future__ import annotations

import signal
import time
from datetime import datetime

import numpy as np

from .commander import Commander
from .config import WS_ROOT, TeleopConfig
from .filters import OneEuro
from .gripper import HandSwitch
from .keyboard import Keyboard
from .retarget import Retargeter
from .robot_spec import JOINT_NAMES, LEFT_GRIPPER_SLOT, NUM_HAND, NUM_JOINTS, RIGHT_GRIPPER_SLOT
from .smpl_stream import SmplStream
from .walk import WalkFollower

SHOW = ["Waist_Yaw", "Waist_Pitch", "Shoulder_Pitch_L", "Shoulder_Roll_L", "Elbow_Pitch_L",
        "Shoulder_Pitch_R", "Shoulder_Roll_R", "Elbow_Pitch_R"]
IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def log(msg: str) -> None:
    print(f"\r\033[K[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Teleop:
    def __init__(self, cfg: TeleopConfig, publish: bool, no_robot: bool = False, use_hand: bool = True):
        if publish and no_robot:
            raise ValueError("--enable-publish needs the robot")
        self.cfg = cfg
        self.publish = publish
        self.no_robot = no_robot
        self.use_hand = use_hand
        self.quit = False
        self.force = False
        self.rows: dict[str, list] = {}

    # ------------------------------------------------------------------ helpers
    def _sigint(self, *_):
        if self.quit:
            self.force = True
            raise KeyboardInterrupt
        log("[STOP] Ctrl+C - returning to the initial pose (Ctrl+C again = stop now)")
        self.quit = True

    def _add(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(np.asarray(v, dtype=np.float64))

    def _save_log(self):
        if not self.rows:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = WS_ROOT / "logs" / f"teleop_{stamp}{'' if self.publish else '_dry'}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **{k: np.stack(v) for k, v in self.rows.items()})
        log(f"log saved: {path}")

    def _control_state_ok(self, io, wait: float = 0.0) -> bool:
        required = list(self.cfg.robot.get("required_control_state") or [])
        if not required or io is None:
            return True
        t_end = time.monotonic() + wait
        st = io.control_state()
        while st is None and time.monotonic() < t_end:
            time.sleep(0.05)
            st = io.control_state()
        if st in required:
            return True
        msg = f"robot control state is {st or 'unknown'}, required {required} (switch the robot to that mode first)"
        if self.publish:
            log("REFUSE: " + msg)
            return False
        if wait:
            log("WARN (dry run continues): " + msg)
        return True

    # --------------------------------------------------------------------- main
    def run(self, addr: str | None = None, max_duration: float = 0.0) -> int:
        cfg = self.cfg
        signal.signal(signal.SIGINT, self._sigint)
        print(cfg.describe(), flush=True)
        log("REAL ROBOT COMMAND ENABLED" if self.publish else
            "DRY RUN - nothing is published" + (" (no robot)" if self.no_robot else ""))

        stream = SmplStream(addr or cfg.stream["addr"], cfg.stream["topic"]).start()
        io = None
        commander = None
        try:
            if not self.no_robot:
                from .robot_io import RobotIO
                io = RobotIO(cfg.robot["domain_id"], cfg.robot["namespace"], use_hand=self.use_hand)
                io.start()
                if not io.wait_ready(5.0):
                    log(f"robot not ready: lowstate_age={io.lowstate_age():.2f}s hand_age={io.hand_age():.2f}s")
                    return 1
                if not self._control_state_ok(io, wait=2.0):
                    return 1
                st = io.state()
                init_body = st["joint_pos_pjs"].copy()
                init_hand = st.get("hand_joint_pos", np.zeros(NUM_HAND)).copy()
                log(f"robot control state: {io.control_state()}")
            else:
                init_body = np.zeros(NUM_JOINTS)
                init_hand = np.zeros(NUM_HAND)

            wcfg = cfg.raw.get("walk") or {}
            self.walk_enabled = bool(wcfg.get("enabled", False)) and not cfg.drives_legs
            if self.publish:
                io.open_publishers(walk=self.walk_enabled)
            hand_slots = [LEFT_GRIPPER_SLOT, RIGHT_GRIPPER_SLOT] if self.use_hand else []
            commander = Commander(
                io, cfg.active, cfg.kp, cfg.kd, cfg.vmax, cfg.lo, cfg.hi,
                command_hz=cfg.control["command_hz"], hand_hz=cfg.control["hand_hz"],
                max_hand_vel=cfg.control["max_hand_vel"], hand_slots=hand_slots, publish=self.publish)
            commander.reset(init_body, init_hand)
            commander.start()
            log(f"holding the initial pose, commander @{cfg.control['command_hz']:.0f} Hz")

            log(f"waiting for SMPL stream {stream.addr} ...")
            while stream.age() > 0.5 and not self.quit:
                time.sleep(0.1)
            if self.quit:
                return 0
            log(f"stream up ({stream.rate():.0f} Hz). Stand in view, then ENTER = engage, q = quit. "
                f"Grippers: {cfg.raw['gripper']['mode']} mode (fist = close, open hand = open; g = switch to z/x keys)")

            try:
                self._loop(stream, io, commander, init_body, init_hand, max_duration)
            finally:
                self._walk_zero(io)

            if not self.force:
                T = float(cfg.control["return_s"])
                log(f"RETURN to the initial pose ({T:.1f}s)")
                commander.set_target(init_body, init_hand if self.use_hand else None, T)
                commander.wait_reached(T + 5.0, tol=2e-3)
                time.sleep(0.5)
                log("done")
            return 0
        except KeyboardInterrupt:
            log("force stop")
            return 130
        finally:
            if commander is not None:
                commander.stop()
            stream.stop()
            self._save_log()
            if io is not None:
                io.stop()

    def _walk_zero(self, io, n: int = 10):
        if self.publish and io is not None and getattr(self, "walk_enabled", False):
            for _ in range(n):
                io.publish_twist(0.0, 0.0, 0.0)
                time.sleep(0.02)

    def _loop(self, stream, io, commander, init_body, init_hand, max_duration):
        cfg = self.cfg
        act = np.array(cfg.active)
        period = 1.0 / float(cfg.control["rate_hz"])
        ramp = float(cfg.control["engage_ramp_s"])
        hold_after = float(cfg.stream["hold_after"])
        pause_after = float(cfg.stream["pause_after"])
        max_track = float(cfg.safety["max_tracking_error"])
        max_state_age = float(cfg.safety["max_state_age"])

        retarget = Retargeter()
        filt = OneEuro(**cfg.filter)
        state = "HOLD"
        hold_pose = init_body.copy()
        hand_goal = init_hand.copy()
        g = cfg.raw["gripper"]
        grip_mode = g["mode"]
        open_q, close_q = float(g["open_q"]), float(g["close_q"])
        switches = {slot: HandSwitch(g["close_above"], g["open_below"], g["hold_s"], closed=init_hand[slot] > 0.5)
                    for slot in (LEFT_GRIPPER_SLOT, RIGHT_GRIPPER_SLOT)}
        closure = np.full(2, np.nan)            # [left, right] from the camera
        warned_no_closure = False
        q_person = None
        last_index = None
        t_engage = 0.0
        t0 = time.monotonic()
        next_t = t0
        last_print = 0.0
        wcfg = cfg.raw.get("walk") or {}
        walker = WalkFollower(wcfg) if self.walk_enabled else None
        walk_on = bool(wcfg.get("start_on", True))
        root_p, root_h = None, None
        target_id, engaged_target = None, None
        last_good = None                          # recv time of the last frame showing the locked person
        age = float("inf")                        # age of the last good pose (set each loop below)
        vel = np.zeros(3)
        if walker is not None:
            log(f"walking: {'ON' if walk_on else 'OFF'} (w = toggle). Step away from your spot / turn to walk, "
                f"come back to stop. dead zone {walker.dz_x:.2f} m / {np.degrees(walker.dz_yaw):.0f} deg, "
                f"max {walker.vx_max:.2f} m/s")

        with Keyboard() as kb:
            if not kb.enabled:
                log("WARN: stdin is not a terminal - keys disabled (ENTER/z/x/q unavailable)")
            while not self.quit:
                now = time.monotonic()
                if max_duration and now - t0 > max_duration:
                    log("max duration reached")
                    break

                # ---- keys
                key = kb.get()
                if key == "q":
                    self.quit = True
                    break
                if key == "ENTER":
                    if state == "HOLD":
                        if q_person is None or age > hold_after:
                            log("cannot engage: no fresh pose of the locked person")
                        else:
                            hold_pose = commander.command()[0]
                            t_engage, state = now, "ENGAGE"
                            for sw in switches.values():
                                sw.reset_timer()
                            engaged_target = target_id
                            log(f"ENGAGE (blending in over {ramp:.1f}s)"
                                + (f", following target #{target_id}" if target_id is not None else ""))
                            if walker is not None:
                                if walker.set_home(root_p, root_h):
                                    log("walk home set here - this spot and facing = stand still")
                                else:
                                    log("WARN: no person position/heading in the stream - walking disabled "
                                        "(update gpu_server/run_fastsam_multiview.py on the GPU PC)")
                    else:
                        hold_pose, state = commander.command()[0], "HOLD"
                        log("PAUSE - holding the current pose")
                if key == "w" and walker is not None:
                    walk_on = not walk_on
                    log(f"walking {'ON' if walk_on else 'OFF'}")
                if key == "h" and walker is not None and state != "HOLD":
                    log("walk home reset here" if walker.set_home(root_p, root_h) else "cannot reset home: no position")
                if key == "g" and self.use_hand:
                    grip_mode = "key" if grip_mode == "hand" else "hand"
                    for slot, sw in switches.items():
                        sw.closed = hand_goal[slot] > 0.5
                    log(f"gripper mode -> {grip_mode}" + (" (z / x toggle)" if grip_mode == "key" else
                                                          " (fist = close, open hand = open)"))
                if key in ("z", "x") and self.use_hand:
                    if grip_mode == "hand":
                        grip_mode = "key"
                        log("gripper mode -> key (g = back to hand control)")
                    slot = LEFT_GRIPPER_SLOT if key == "z" else RIGHT_GRIPPER_SLOT
                    hand_goal[slot] = open_q if hand_goal[slot] > 0.5 else close_q
                    log(f"{'left' if key == 'z' else 'right'} gripper -> "
                        f"{'CLOSE' if hand_goal[slot] > 0.5 else 'OPEN'}")

                # ---- person pose
                fr = stream.latest()
                age = stream.age()
                new = fr is not None and fr.frame_index != last_index
                if new:
                    last_index = fr.frame_index
                    tok, tid = fr.extra.get("target_ok"), fr.extra.get("target_id")
                    if tid is not None:
                        target_id = int(np.ravel(tid)[0])
                    if tok is not None and float(np.ravel(tok)[0]) < 0.5:
                        new = False                   # locked person not visible: stale, like a lost stream
                    else:
                        last_good = fr.recv_time
                if last_good is not None:
                    age = max(age, time.monotonic() - last_good)
                if (state != "HOLD" and target_id is not None and engaged_target is not None
                        and target_id != engaged_target):
                    hold_pose, state = commander.command()[0], "HOLD"
                    if walker is not None:
                        walker.stop()
                    log(f"PAUSE - camera locked a different person (target #{target_id}); ENTER to re-engage")
                if new:
                    q_rt = retarget(fr.smpl_pose).q
                    q_person = filt(cfg.shape(q_rt), fr.recv_time)
                    rp, rh = fr.extra.get("root_pos"), fr.extra.get("root_heading")
                    root_p = None if rp is None else np.asarray(rp, float).reshape(-1)[:3]
                    root_h = None if rh is None else float(np.asarray(rh, float).reshape(-1)[0])
                    hc = fr.extra.get("hand_closure")
                    closure = np.asarray(hc, float).reshape(-1)[:2] if hc is not None else np.full(2, np.nan)
                    if hc is None and grip_mode == "hand" and self.use_hand and not warned_no_closure:
                        warned_no_closure = True
                        log("WARN: the stream has no hand_closure (start the publisher with "
                            "gpu_server/run_fastsam_multiview.py) - grippers stay put; g = key control")
                    # grippers follow the hand only while the arms follow the person
                    if grip_mode == "hand" and self.use_hand and state in ("ENGAGE", "TRACK"):
                        for k, (slot, side) in enumerate(((LEFT_GRIPPER_SLOT, "left"), (RIGHT_GRIPPER_SLOT, "right"))):
                            if switches[slot].update(closure[k], fr.recv_time):
                                hand_goal[slot] = close_q if switches[slot].closed else open_q
                                log(f"{side} gripper -> {'CLOSE' if switches[slot].closed else 'OPEN'} "
                                    f"(hand closure {closure[k]:.2f})")
                if state != "HOLD" and age > pause_after:
                    hold_pose, state = commander.command()[0], "HOLD"
                    log(f"PAUSE - camera stream lost ({age:.1f}s); ENTER to re-engage")

                # ---- robot checks
                meas = None
                if io is not None:
                    if io.lowstate_age() > max_state_age:
                        if state != "HOLD":
                            hold_pose, state = commander.command()[0], "HOLD"
                        log(f"PAUSE - LowState stale ({io.lowstate_age():.2f}s)")
                    else:
                        meas = io.state()["joint_pos_pjs"]
                    if self.publish and not self._control_state_ok(io):
                        log("ABORT: control mode changed - stopping command streaming")
                        self.force = True
                        break
                    if self.publish and meas is not None:
                        err = np.abs(commander.command()[0][act] - meas[act])
                        if err.max() > max_track:
                            j = JOINT_NAMES[act[int(np.argmax(err))]]
                            log(f"PAUSE - tracking error {err.max():.2f} rad on {j}; holding the measured pose")
                            hold_pose = meas.copy()
                            commander.reset(meas, None)
                            state = "HOLD"

                # ---- target
                target = hold_pose.copy()
                if state in ("ENGAGE", "TRACK") and q_person is not None and age <= hold_after:
                    goal = hold_pose.copy()
                    goal[act] = np.where(np.isnan(q_person[act]), hold_pose[act], q_person[act])
                    if state == "ENGAGE":
                        a = min((now - t_engage) / ramp, 1.0)
                        target = hold_pose + a * (goal - hold_pose)
                        if a >= 1.0:
                            state = "TRACK"
                            log("TRACK - the robot follows you (ENTER = pause)")
                    else:
                        target = goal
                    commander.set_target(target, hand_goal, period)
                elif state == "HOLD":
                    commander.set_target(hold_pose, hand_goal, period)
                else:  # engaged but the stream is briefly late: keep the last goal, still update hands
                    commander.set_target(None, hand_goal, period)

                if walker is not None:
                    vel = walker.update(root_p, root_h, now, period,
                                        enabled=walk_on and state == "TRACK" and age <= hold_after)
                    if self.publish and io is not None:
                        io.publish_twist(*vel)

                cmd_b, cmd_h = commander.command()
                self._add(walk_vel=vel, walk_offset=walker.offset if walker is not None else np.zeros(3),
                          t=now - t0, state={"HOLD": 0, "ENGAGE": 1, "TRACK": 2}[state], stream_age=min(age, 99),
                          q_person=q_person if q_person is not None else np.full(NUM_JOINTS, np.nan),
                          cmd=cmd_b, cmd_hand=cmd_h, hand_closure=closure,
                          meas=meas if meas is not None else np.full(NUM_JOINTS, np.nan))

                if now - last_print > 0.5:
                    last_print = now
                    vals = " ".join(f"{n.replace('Shoulder_', 'S').replace('Elbow_Pitch', 'EP').replace('Waist_', 'W')}"
                                    f"={cmd_b[IDX[n]]:+.2f}" for n in SHOW)
                    grip = ""
                    if self.use_hand:
                        gl = "C" if hand_goal[LEFT_GRIPPER_SLOT] > 0.5 else "O"
                        gr = "C" if hand_goal[RIGHT_GRIPPER_SLOT] > 0.5 else "O"
                        grip = f" | grip[{grip_mode}] L {gl} {closure[0]:.2f}  R {gr} {closure[1]:.2f}"
                    walk = ""
                    if walker is not None:
                        o = walker.offset
                        walk = (f" | walk[{'on' if walk_on else 'off'}] d {o[0]:+.2f} {o[1]:+.2f}m "
                                f"{np.degrees(o[2]):+.0f}deg v {vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f}")
                    print(f"\r\033[K{state:6s} cam {stream.rate():4.1f}Hz age {min(age, 9.9):.2f}s | {vals}{grip}{walk}",
                          end="", flush=True)

                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
        print()
