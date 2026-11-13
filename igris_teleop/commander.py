"""300 Hz LowCmd / 50 Hz HandCmd streamer (adapted from igris_demo_ws).

Targets arrive at ~50 Hz from the teleop loop; this thread interpolates toward
them, applies a per-joint velocity limit and the joint limits, and publishes.
Joints not driven by teleop get kp = kd = 0 (the walk controller owns the legs).
Parallel-link groups that contain a driven joint are commanded in PJS.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .robot_spec import HAND_MOTOR_IDS, NUM_HAND, NUM_JOINTS, PARALLEL_LINK_JOINTS


class _Segment:
    """Linear interpolation from `start` to `goal` over `duration`, plus per-joint rate limit."""

    def __init__(self, n: int, vmax, lo, hi):
        self.cmd = np.zeros(n)
        self.start = np.zeros(n)
        self.goal = np.zeros(n)
        self.t0 = 0.0
        self.T = 1e-3
        self.vmax = np.broadcast_to(np.asarray(vmax, float), (n,)).copy()
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def reset(self, x: np.ndarray) -> None:
        self.cmd[:] = x
        self.start[:] = x
        self.goal[:] = x

    def set(self, goal: np.ndarray, duration: float, now: float) -> None:
        g = np.where(np.isnan(goal), self.goal, goal)
        g = np.clip(g, self.lo, self.hi)
        self.start[:] = self.cmd
        self.goal[:] = g
        self.t0, self.T = now, max(duration, 1e-3)

    def tick(self, now: float, dt: float, mask: np.ndarray) -> None:
        a = min((now - self.t0) / self.T, 1.0)
        desired = self.start + a * (self.goal - self.start)
        step = np.clip(desired - self.cmd, -self.vmax * dt, self.vmax * dt)
        self.cmd[mask] += step[mask]

    def reached(self, mask: np.ndarray, tol: float) -> bool:
        return bool(np.all(np.abs(self.goal[mask] - self.cmd[mask]) < tol))


class Commander:
    def __init__(self, io, active: list[int], kp, kd, vmax, lo, hi, *,
                 command_hz=300.0, hand_hz=50.0, max_hand_vel=3.0, hand_slots=(), publish=False):
        self.io = io
        self.publish = publish
        self.dt = 1.0 / float(command_hz)
        self.hand_every = max(1, int(round(command_hz / float(hand_hz))))

        self.body_mask = np.zeros(NUM_JOINTS, bool)
        self.body_mask[list(active)] = True
        self.hand_mask = np.zeros(NUM_HAND, bool)
        self.hand_mask[list(hand_slots)] = True
        self.use_hand = bool(hand_slots)

        self.kp = np.where(self.body_mask, np.asarray(kp, float), 0.0)
        self.kd = np.where(self.body_mask, np.asarray(kd, float), 0.0)
        self.body = _Segment(NUM_JOINTS, vmax, lo, hi)
        self.hand = _Segment(NUM_HAND, max_hand_vel, np.zeros(NUM_HAND), np.ones(NUM_HAND))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ticks = 0
        if publish:
            import igris_c_sdk as igc
            self.igc = igc
            self._lowcmd = igc.LowCmd()
            self.kin_modes = [igc.KinematicMode.PJS if any(self.body_mask[j] for j in pair)
                              else igc.KinematicMode.MS for pair in PARALLEL_LINK_JOINTS]

    # ----------------------------------------------------------------- control
    def reset(self, body: np.ndarray, hand: np.ndarray | None) -> None:
        with self._lock:
            # the initial pose may sit slightly outside the soft limits: start exactly there
            self.body.reset(body)
            if hand is not None:
                self.hand.reset(np.clip(hand, 0.0, 1.0))

    def set_target(self, body: np.ndarray | None, hand: np.ndarray | None, duration: float) -> None:
        now = time.monotonic()
        with self._lock:
            if body is not None:
                self.body.set(np.asarray(body, float), duration, now)
            if hand is not None and self.use_hand:
                self.hand.set(np.asarray(hand, float), duration, now)

    def command(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self.body.cmd.copy(), self.hand.cmd.copy()

    def wait_reached(self, timeout: float, tol: float = 1e-3) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._lock:
                ok = self.body.reached(self.body_mask, tol) and (
                    not self.use_hand or self.hand.reached(self.hand_mask, tol))
            if ok:
                return True
            time.sleep(0.01)
        return False

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="commander", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print("[commander] WARNING: streaming thread did not stop", flush=True)

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            next_t += self.dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
            now = time.monotonic()
            with self._lock:
                self.body.tick(now, self.dt, self.body_mask)
                if self.use_hand:
                    self.hand.tick(now, self.dt, self.hand_mask)
                body = self.body.cmd.copy()
                hand = self.hand.cmd.copy()
            if self.publish:
                self._write_lowcmd(body)
                if self.use_hand and self.ticks % self.hand_every == 0:
                    self._write_handcmd(hand)
            self.ticks += 1

    def _write_lowcmd(self, q: np.ndarray) -> None:
        cmd = self._lowcmd
        cmd.kinematic_modes(self.kin_modes)      # list setter (see sdk_gui_client.py)
        motors = cmd.motors()
        for i in range(NUM_JOINTS):
            m = motors[i]
            m.id(i)
            m.q(float(q[i]))
            m.dq(0.0)
            m.tau(0.0)
            m.kp(float(self.kp[i]))
            m.kd(float(self.kd[i]))
        self.io.lowcmd_pub.write(cmd)

    def _write_handcmd(self, q: np.ndarray) -> None:
        motors = []
        for mid, v in zip(HAND_MOTOR_IDS, q):
            m = self.igc.MotorCmd()
            m.id(mid)
            m.q(float(np.clip(v, 0.0, 1.0)))
            motors.append(m)
        cmd = self.igc.HandCmd()
        cmd.motor_cmd(motors)
        self.io.handcmd_pub.write(cmd)
