"""Person movement -> walk velocity (cmd_vel) for the robot's walk controller.

Low-Level Walkmode keeps the balance; this only tells the walk controller where to go,
like a joystick whose stick is the person's position:

  ENTER (engage) stores the person's HOME spot and facing direction.
  Step forward / back / sideways from HOME  -> walk forward / back / sideways
  Turn the body (hips + shoulders) left / right  -> turn left / right
  Come back to the HOME spot and face forward     -> stop

Positions come from the GPU publisher (`root_pos` [m], `root_heading` [rad], main
camera, gravity-aligned floor frame). Everything is expressed in the person's HOME
frame, so it does not matter where the camera stands.

Safety: dead zones, the person must stay outside them for `hold_s` before walking
starts, velocity limits, acceleration limits, and zero velocity whenever the
position / heading is missing or implausible (|turn| > max_turn).
"""

from __future__ import annotations

import math

import numpy as np


def _dead(x, dz, gain, vmax):
    if abs(x) <= dz:
        return 0.0
    return float(np.clip(gain * (abs(x) - dz), 0.0, vmax) * math.copysign(1.0, x))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class WalkFollower:
    def __init__(self, cfg: dict):
        self.dz_x = float(cfg.get("deadzone_fwd_m", 0.20))
        self.dz_y = float(cfg.get("deadzone_side_m", 0.20))
        self.dz_yaw = math.radians(float(cfg.get("deadzone_turn_deg", 20.0)))
        self.max_turn = math.radians(float(cfg.get("max_turn_deg", 75.0)))
        self.gain_lin = float(cfg.get("gain_lin", 1.0))      # (m/s) per m beyond the dead zone
        self.gain_yaw = float(cfg.get("gain_yaw", 1.0))      # (rad/s) per rad beyond the dead zone
        self.vx_max = float(cfg.get("max_vx", 0.3))
        self.vy_max = float(cfg.get("max_vy", 0.15))
        self.wz_max = float(cfg.get("max_wz", 0.4))
        self.acc_lin = float(cfg.get("acc_lin", 0.5))        # m/s^2
        self.acc_yaw = float(cfg.get("acc_yaw", 1.0))        # rad/s^2
        self.hold_s = float(cfg.get("hold_s", 0.3))
        self.alpha = float(cfg.get("smooth", 0.3))           # EMA on position / heading (0..1, 1 = none)
        self.home = None                                     # (p_xy, heading)
        self.p = None
        self.h = None
        self.v = np.zeros(3)
        self._out_since = None
        self.offset = np.zeros(3)                            # last (dx, dy, dyaw) in the home frame

    def set_home(self, p, h) -> bool:
        p = None if p is None else np.asarray(p, float).reshape(-1)[:2]
        if p is None or h is None or not np.all(np.isfinite(p)) or not math.isfinite(h):
            self.home = None
            return False
        self.home = (p.copy(), float(h))
        self.p, self.h = p.copy(), float(h)
        self._out_since = None
        return True

    def stop(self):
        self.v[:] = 0.0
        self._out_since = None

    def update(self, p, h, now: float, dt: float, enabled: bool) -> np.ndarray:
        """Returns (vx, vy, wz) to send. `enabled` False -> decelerate to zero."""
        want = np.zeros(3)
        ok = (enabled and self.home is not None and p is not None and h is not None)
        if ok:
            p = np.asarray(p, float).reshape(-1)[:2]
            ok = bool(np.all(np.isfinite(p)) and math.isfinite(h))
        if ok:
            self.p = p if self.p is None else self.p + self.alpha * (p - self.p)
            dh = _wrap(h - self.h) if self.h is not None else 0.0
            self.h = h if self.h is None else _wrap(self.h + self.alpha * dh)
            p0, h0 = self.home
            d = self.p - p0
            dx = d[0] * math.cos(h0) + d[1] * math.sin(h0)
            dy = -d[0] * math.sin(h0) + d[1] * math.cos(h0)
            dyaw = _wrap(self.h - h0)
            self.offset = np.array([dx, dy, dyaw])
            if abs(dyaw) > self.max_turn:            # heading estimate implausible (or person turned away)
                ok = False
            else:
                want = np.array([_dead(dx, self.dz_x, self.gain_lin, self.vx_max),
                                 _dead(dy, self.dz_y, self.gain_lin, self.vy_max),
                                 _dead(dyaw, self.dz_yaw, self.gain_yaw, self.wz_max)])
        if ok and np.any(want):
            if self._out_since is None:
                self._out_since = now
            if now - self._out_since < self.hold_s:
                want[:] = 0.0
        else:
            self._out_since = None
            if not ok:
                want[:] = 0.0
        acc = np.array([self.acc_lin, self.acc_lin, self.acc_yaw]) * dt
        self.v = self.v + np.clip(want - self.v, -acc, acc)
        self.v[np.abs(self.v) < 1e-4] = 0.0
        return self.v.copy()
