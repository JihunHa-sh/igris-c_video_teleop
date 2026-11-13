"""SMPL body pose -> IGRIS-C upper-body joint angles (PJS, radians).

Plays the role SONIC's learned SMPL encoder plays for the Unitree G1, but as an
explicit, inspectable kinematic mapping for IGRIS (no IGRIS SONIC policy exists).
Legs are retargeted the same way (thigh / shank directions + foot orientation),
but are only COMMANDED when the config drives them: in Low-Level Walkmode the
walk controller owns the legs, so by default the leg angles are for the viewer.

Frames (all right-handed)
  SMPL canonical : x = left,    y = up,   z = forward
  IGRIS (URDF)   : x = forward, y = left, z = up        robot = C @ smpl

IGRIS kinematics (assets/urdf/igris_c.urdf; signs cross-checked with the real
robot's resting pose: shoulder roll L +0.12 / R -0.10, elbow -0.32)
  base_link is the TORSO. Waist chain torso -> pelvis: Rz(yaw) Rx(roll) Ry(pitch)
  arm chain from torso: Ry(sh_pitch) Rx(sh_roll) Rz(sh_yaw) | Ry(elbow) | Rz(wr_yaw) Rx(wr_roll) Ry(wr_pitch)
  zero pose = arms hanging straight down; SMPL zero = T-pose.
  leg chain from the pelvis: Ry(hip_pitch) Rx(hip_roll) Rz(hip_yaw) | Ry(knee) | Ry(ankle_pitch) Rx(ankle_roll)
  zero pose = legs straight down, feet flat (same as SMPL zero). Knee > 0 = flexed.

Arms are solved geometrically so the robot's upper-arm and forearm point where
the person's do (the robot elbow is 1-DoF, the SMPL elbow is not); the SMPL
upper-arm twist is used only when the elbow is nearly straight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import smpl
from .robot_spec import JOINT_NAMES

C = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # robot = C @ smpl
IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def to_robot(R_smpl: np.ndarray) -> np.ndarray:
    return C @ R_smpl @ C.T


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------- Euler decompositions
# Each returns angles (a, b, c) with R = R_first(a) R_second(b) R_third(c). Near the
# gimbal lock (|middle| -> 90 deg) the first angle is held at `prev_first` so the
# solution stays continuous (the robot shoulder has the same physical singularity).

def euler_yxz(R, prev_first=0.0, eps=1e-3):
    """R = Ry(a) Rx(b) Rz(c)."""
    sb = -R[1, 2]
    b = np.arcsin(np.clip(sb, -1.0, 1.0))
    if np.sqrt(R[0, 2] ** 2 + R[2, 2] ** 2) > eps:
        a = np.arctan2(R[0, 2], R[2, 2])
        c = np.arctan2(R[1, 0], R[1, 1])
    else:
        a = prev_first
        rest = rot_x(b).T @ rot_y(a).T @ R
        c = np.arctan2(rest[1, 0], rest[0, 0])
    return a, b, c


def euler_zxy(R, prev_first=0.0, eps=1e-3):
    """R = Rz(a) Rx(b) Ry(c)."""
    sb = R[2, 1]
    b = np.arcsin(np.clip(sb, -1.0, 1.0))
    if np.sqrt(R[0, 1] ** 2 + R[1, 1] ** 2) > eps:
        a = np.arctan2(-R[0, 1], R[1, 1])
        c = np.arctan2(-R[2, 0], R[2, 2])
    else:
        a = prev_first
        rest = rot_x(b).T @ rot_z(a).T @ R
        c = np.arctan2(rest[0, 2], rest[0, 0])
    return a, b, c


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ------------------------------------------------------------------------ retargeter
@dataclass
class RetargetResult:
    q: np.ndarray                      # (31,) PJS angles; NaN for joints not produced
    debug: dict = field(default_factory=dict)


class Retargeter:
    """Stateless mapping except for Euler-continuity memory (call reset() between sessions)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._prev = {"L": 0.0, "R": 0.0, "waist": 0.0, "wL": 0.0, "wR": 0.0, "neck": 0.0,
                      "legL": 0.0, "legR": 0.0}

    # ---- waist: torso orientation relative to the pelvis
    def _waist(self, Rl):
        R_pt = to_robot(smpl.chain(Rl, "spine1", "spine2", "spine3"))  # torso in pelvis frame
        # Robot chain goes torso -> pelvis, so the joint rotation is the inverse.
        yaw, roll, pitch = euler_zxy(R_pt.T, self._prev["waist"])
        self._prev["waist"] = yaw
        return yaw, roll, pitch

    # ---- arms
    def _arm(self, Rl, side):
        s = 1.0 if side == "L" else -1.0
        pre = "left" if side == "L" else "right"
        O = rot_x(s * np.pi / 2)                          # robot arm frame at SMPL T-pose
        U = to_robot(smpl.chain(Rl, f"{pre}_collar", f"{pre}_shoulder"))   # upper arm, torso frame
        E = to_robot(Rl[smpl.J[f"{pre}_elbow"]])
        W = to_robot(Rl[smpl.J[f"{pre}_wrist"]])
        rest = np.array([0.0, s, 0.0])                     # SMPL rest bone direction (robot coords)
        u = U @ rest                                       # upper-arm direction (shoulder -> elbow)
        f = U @ E @ rest                                   # forearm direction (elbow -> wrist)

        cos_e = float(np.clip(u @ f, -1.0, 1.0))
        elbow = -np.arccos(cos_e)                          # robot elbow: 0 = straight, negative = flexed
        sin_e = np.sqrt(max(0.0, 1.0 - cos_e ** 2))

        # Link frame: z points back up the upper arm, x toward the forearm (elbow plane).
        z_l = -u
        R_rot = U @ O                                      # rotation-based guess (carries upper-arm twist)
        x_twist = _unit(R_rot[:, 0] - (R_rot[:, 0] @ z_l) * z_l)
        x_bend = _unit(f - (f @ u) * u) if sin_e > 1e-6 else x_twist
        w = np.clip((sin_e - 0.10) / 0.25, 0.0, 1.0)       # elbow < ~6 deg: twist; > ~20 deg: geometry
        x_l = _unit(w * x_bend + (1.0 - w) * x_twist)
        y_l = np.cross(z_l, x_l)
        R_link = np.column_stack([x_l, y_l, z_l])

        pitch, roll, yaw = euler_yxz(R_link, self._prev[side])
        self._prev[side] = pitch

        # Wrist: hand orientation relative to the forearm link.
        H = U @ E @ W @ O                                  # SMPL hand frame in robot arm-link convention
        R_fore = R_link @ rot_y(elbow)
        wy, wr, wp = euler_zxy(R_fore.T @ H, self._prev["w" + side])
        self._prev["w" + side] = wy
        return (pitch, roll, yaw, elbow, wy, wr, wp), {"u": u, "f": f, "R_link": R_link}

    # ---- legs (in the pelvis frame; the hips hang off the pelvis, Link_Waist_Pitch)
    def _leg(self, Rl, side):
        pre = "left" if side == "L" else "right"
        Hh = to_robot(Rl[smpl.J[f"{pre}_hip"]])            # thigh, pelvis frame
        K = to_robot(Rl[smpl.J[f"{pre}_knee"]])
        A = to_robot(Rl[smpl.J[f"{pre}_ankle"]])
        down = np.array([0.0, 0.0, -1.0])                  # rest bone direction (robot coords)
        t = Hh @ down                                      # thigh direction (hip -> knee)
        sh = Hh @ K @ down                                 # shank direction (knee -> ankle)

        cos_k = float(np.clip(t @ sh, -1.0, 1.0))
        knee = np.arccos(cos_k)                            # robot knee: 0 = straight, > 0 = flexed
        sin_k = np.sqrt(max(0.0, 1.0 - cos_k ** 2))

        # Link frame: z back up the thigh, x forward (a flexed knee sends the shank to -x).
        z_l = -t
        x_twist = _unit(Hh[:, 0] - (Hh[:, 0] @ z_l) * z_l)  # thigh twist from the SMPL hip rotation
        x_bend = -_unit(sh - (sh @ t) * t) if sin_k > 1e-6 else x_twist
        # Trust the bend plane only when the knee is clearly flexed AND it bends the
        # natural way (a "forward" bend is estimation noise and would flip the hip yaw).
        w = np.clip((sin_k - 0.10) / 0.25, 0.0, 1.0) * np.clip((x_bend @ x_twist - 0.3) / 0.4, 0.0, 1.0)
        x_l = _unit(w * x_bend + (1.0 - w) * x_twist)
        if w < 1.0:   # bend plane not used -> knee from the shank angle in the twist plane
            knee = max(0.0, np.arctan2(-(sh @ x_l), -(sh @ z_l)))
        R_link = np.column_stack([x_l, np.cross(z_l, x_l), z_l])

        pitch, roll, yaw = euler_yxz(R_link, self._prev["leg" + side])
        self._prev["leg" + side] = pitch

        # Ankle (2-DoF): foot orientation relative to the shank, R = Ry(pitch) Rx(roll).
        rel = (R_link @ rot_y(knee)).T @ (Hh @ K @ A)
        a_pitch = np.arctan2(-rel[2, 0], rel[0, 0])
        a_roll = np.arctan2(-rel[1, 2], rel[1, 1])
        return (pitch, roll, yaw, knee, a_pitch, a_roll), {"t": t, "s": sh}

    # ---- neck: head orientation relative to the torso
    def _neck(self, Rl):
        R_h = to_robot(smpl.chain(Rl, "neck", "head"))
        yaw, _roll, pitch = euler_zxy(R_h, self._prev["neck"])
        self._prev["neck"] = yaw
        return yaw, pitch

    def __call__(self, smpl_pose: np.ndarray) -> RetargetResult:
        Rl = smpl.local_rotations(smpl_pose)
        q = np.full(len(JOINT_NAMES), np.nan)
        q[[IDX["Waist_Yaw"], IDX["Waist_Roll"], IDX["Waist_Pitch"]]] = self._waist(Rl)
        dbg = {}
        for side, suf in (("L", "_L"), ("R", "_R")):
            angles, d = self._arm(Rl, side)
            names = ["Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch",
                     "Wrist_Yaw", "Wrist_Roll", "Wrist_Pitch"]
            q[[IDX[n + suf] for n in names]] = angles
            dbg[side] = d
        leg_names = ["Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll"]
        for side, suf in (("L", "_L"), ("R", "_R")):
            angles, d = self._leg(Rl, side)
            q[[IDX[n + suf] for n in leg_names]] = angles
            dbg["leg" + side] = d
        q[[IDX["Neck_Yaw"], IDX["Neck_Pitch"]]] = self._neck(Rl)
        return RetargetResult(q=q, debug=dbg)
