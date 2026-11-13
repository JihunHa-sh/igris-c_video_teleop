"""Minimal URDF forward kinematics (revolute/fixed joints, origin xyz/rpy, axis).

Used to check the retargeter (robot bone directions vs. the person's) and to draw
the robot skeleton in the viewer. Fixed joints are kept fixed; joint values are
looked up by SDK name (Joint_Waist_Yaw -> Waist_Yaw, Joint_Elbow_Pitch_Left -> Elbow_Pitch_L).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .robot_spec import JOINT_NAMES

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "assets" / "urdf" / "igris_c.urdf"


def urdf_to_sdk_name(name: str) -> str | None:
    n = name.removeprefix("Joint_")
    n = n.replace("_Left", "_L").replace("_Right", "_R")
    return n if n in JOINT_NAMES else None


def _axis_angle(axis, a):
    return Rotation.from_rotvec(np.asarray(axis) * a).as_matrix()


class UrdfFK:
    def __init__(self, path: str | Path = DEFAULT_URDF, root: str = "base_link"):
        tree = ET.parse(path).getroot()
        self.joints = []
        for j in tree.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.zeros(3)
            self.joints.append(dict(
                name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                child=j.find("child").get("link"), xyz=xyz,
                R0=Rotation.from_euler("xyz", rpy).as_matrix(), axis=axis,
                sdk=urdf_to_sdk_name(j.get("name")),
            ))
        self.root = root
        self.children: dict[str, list[dict]] = {}
        for j in self.joints:
            self.children.setdefault(j["parent"], []).append(j)

    def link_poses(self, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """q: (31,) PJS angles (NaN treated as 0). Returns {link: (R, p)} in the root (torso) frame."""
        q = np.nan_to_num(np.asarray(q, dtype=np.float64))
        sdk_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
        out = {self.root: (np.eye(3), np.zeros(3))}
        stack = [self.root]
        while stack:
            link = stack.pop()
            R, p = out[link]
            for j in self.children.get(link, []):
                Rj = R @ j["R0"]
                pj = p + R @ j["xyz"]
                if j["type"] in ("revolute", "continuous") and j["sdk"] is not None:
                    Rj = Rj @ _axis_angle(j["axis"], q[sdk_idx[j["sdk"]]])
                out[j["child"]] = (Rj, pj)
                stack.append(j["child"])
        return out

    def positions(self, q, links) -> np.ndarray:
        poses = self.link_poses(q)
        return np.array([poses[l][1] for l in links])


# Points drawn for the robot skeleton (torso frame). Wrist roll link = hand position.
ROBOT_DRAW = {
    "spine": ["Link_Waist_Pitch", "base_link", "Link_Neck_Pitch"],
    "left_arm": ["Link_Shoulder_Pitch_Left", "Link_Elbow_Pitch_Left", "Link_Wrist_Roll_Left", "Left_Hand"],
    "right_arm": ["Link_Shoulder_Pitch_Right", "Link_Elbow_Pitch_Right", "Link_Wrist_Roll_Right", "Right_Hand"],
    "shoulders": ["Link_Shoulder_Pitch_Left", "Link_Shoulder_Pitch_Right"],
    "hips": ["Link_Hip_Pitch_Left", "Link_Waist_Pitch", "Link_Hip_Pitch_Right"],
    "left_leg": ["Link_Hip_Pitch_Left", "Link_Knee_Pitch_Left", "Link_Ankle_Roll_Left"],
    "right_leg": ["Link_Hip_Pitch_Right", "Link_Knee_Pitch_Right", "Link_Ankle_Roll_Right"],
}
