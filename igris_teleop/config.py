"""Load config/teleop.yaml into a small object with resolved joint indices / arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .robot_spec import (
    DEFAULT_KD, DEFAULT_KP, JOINT_GROUPS, JOINT_NAMES, JOINT_POS_MAX, JOINT_POS_MIN, NUM_JOINTS,
)

WS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WS_ROOT / "config" / "teleop.yaml"
IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def joint_indices(spec) -> list[int]:
    out: list[int] = []
    for item in spec or []:
        if item in JOINT_GROUPS:
            out += JOINT_GROUPS[item]
        elif item in IDX:
            out.append(IDX[item])
        else:
            raise ValueError(f"unknown joint/group {item!r}; groups: {list(JOINT_GROUPS)}")
    return sorted(set(out))


def _vel_class(name: str) -> str:
    if name.startswith("Waist"):
        return "waist"
    if name.startswith("Wrist"):
        return "wrist"
    if name.startswith("Neck"):
        return "neck"
    if name.startswith(("Hip", "Knee", "Ankle")):
        return "leg"
    return "arm"


LEG_JOINTS = set(JOINT_GROUPS["left_leg"] + JOINT_GROUPS["right_leg"])


class TeleopConfig:
    def __init__(self, raw: dict, path: Path | None = None):
        self.raw = raw
        self.path = path
        self.stream = raw["stream"]
        self.robot = raw["robot"]
        self.filter = raw["filter"]
        self.control = raw["control"]
        self.safety = raw["safety"]

        self.active = joint_indices(raw["drive"])
        self.drives_legs = bool(LEG_JOINTS & set(self.active))
        required = list(self.robot.get("required_control_state") or [])
        if self.drives_legs and (not required or any(st != "LOW" for st in required)):
            raise ValueError(
                "legs are driven, so the robot must be in Low Level Joint Control only "
                "(robot.required_control_state: [LOW]); in Walkmode the walk controller owns the legs. "
                "Use config/hanger_legs.yaml with the robot on the hanger.")
        for i in self.active:
            if i not in DEFAULT_KP and JOINT_NAMES[i] not in raw["gains"]["kp"]:
                raise ValueError(f"{JOINT_NAMES[i]} has no verified gain; set gains.kp/kd for it")

        self.scale = np.ones(NUM_JOINTS)
        self.offset = np.zeros(NUM_JOINTS)
        self.sign = np.ones(NUM_JOINTS)
        self.lo = JOINT_POS_MIN.astype(float).copy()
        self.hi = JOINT_POS_MAX.astype(float).copy()
        for name, s in (raw.get("shaping") or {}).items():
            i = IDX[name]
            self.scale[i] = float(s.get("scale", 1.0))
            self.offset[i] = float(s.get("offset", 0.0))
            self.sign[i] = float(s.get("sign", 1.0))
            self.lo[i] = max(self.lo[i], float(s.get("limit_min", self.lo[i])))
            self.hi[i] = min(self.hi[i], float(s.get("limit_max", self.hi[i])))
        m = float(raw.get("limit_margin", 0.0))
        self.lo, self.hi = self.lo + m, self.hi - m

        self.kp = np.zeros(NUM_JOINTS)
        self.kd = np.zeros(NUM_JOINTS)
        for i in self.active:
            n = JOINT_NAMES[i]
            self.kp[i] = float(raw["gains"]["kp"].get(n, DEFAULT_KP.get(i, 0.0)))
            self.kd[i] = float(raw["gains"]["kd"].get(n, DEFAULT_KD.get(i, 0.0)))

        mv = self.control["max_vel"]
        self.vmax = np.array([float(mv.get(_vel_class(n), 0.5)) for n in JOINT_NAMES])

    def shape(self, q_retarget: np.ndarray) -> np.ndarray:
        """Retargeted angles -> commanded angles (scale/offset/sign, limits). NaN stays NaN."""
        q = self.sign * self.scale * q_retarget + self.offset
        return np.where(np.isnan(q), np.nan, np.clip(q, self.lo, self.hi))

    def describe(self) -> str:
        lines = [f"config: {self.path}", f"stream: {self.stream['addr']}",
                 "driven joints: " + ", ".join(JOINT_NAMES[i] for i in self.active)]
        if self.drives_legs:
            lines.append("LEGS DRIVEN (Low Level Joint Control, no balance): the robot must hang on the hanger")
        shaped = [f"{JOINT_NAMES[i]} x{self.sign[i] * self.scale[i]:g}{self.offset[i]:+g}"
                  for i in self.active if self.scale[i] != 1 or self.offset[i] or self.sign[i] != 1]
        if shaped:
            lines.append("shaping: " + ", ".join(shaped))
        return "\n".join(lines)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TeleopConfig":
        p = Path(path) if path else DEFAULT_CONFIG
        with open(p) as f:
            raw = yaml.safe_load(f)
        return cls(raw, p)
