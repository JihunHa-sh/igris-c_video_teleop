"""Static IGRIS-C robot description.

Sources:
  - joint names / limits / parallel-link groups: igris_c_sdk_public/python/sdk_gui_client.py
  - KP/KD for waist + arms: igris_ws/src/igris_replay (verified on the real robot)
  - hand motor ids: igris_ws/src/igris_replay/scripts/hand_replay.py (copied from igris_demo_ws)
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 31
NUM_HAND = 12

JOINT_NAMES: tuple[str, ...] = (
    "Waist_Yaw", "Waist_Roll", "Waist_Pitch",
    "Hip_Pitch_L", "Hip_Roll_L", "Hip_Yaw_L", "Knee_Pitch_L", "Ankle_Pitch_L", "Ankle_Roll_L",
    "Hip_Pitch_R", "Hip_Roll_R", "Hip_Yaw_R", "Knee_Pitch_R", "Ankle_Pitch_R", "Ankle_Roll_R",
    "Shoulder_Pitch_L", "Shoulder_Roll_L", "Shoulder_Yaw_L", "Elbow_Pitch_L",
    "Wrist_Yaw_L", "Wrist_Roll_L", "Wrist_Pitch_L",
    "Shoulder_Pitch_R", "Shoulder_Roll_R", "Shoulder_Yaw_R", "Elbow_Pitch_R",
    "Wrist_Yaw_R", "Wrist_Roll_R", "Wrist_Pitch_R",
    "Neck_Yaw", "Neck_Pitch",
)

# Joint-space (PJS) position limits [rad].
JOINT_POS_MIN = np.array((
    -1.57, -0.31, -0.87, -2.0, -0.33, -1.57, 0.0, -0.7, -0.35, -2.0, -2.3, -1.57, 0.0, -0.7,
    -0.35, -3.141, -0.17, -1.57, -2.0, -1.57, -1.221, -0.65, -3.141, -3.14, -1.57, -2.0,
    -1.57, -0.87, -0.65, -1.221, -0.52,
))
JOINT_POS_MAX = np.array((
    1.57, 0.31, 0.28, 0.48, 2.3, 1.57, 2.28, 0.698, 0.349, 0.48, 0.33, 1.57, 2.28, 0.698,
    0.349, 1.047, 3.14, 1.57, 0.0, 1.57, 0.87, 0.65, 1.047, 0.17, 1.57, 0.0, 1.57, 1.221,
    0.65, 1.221, 0.52,
))

# LowCmd.kinematic_modes[i] selects MS / PJS for each parallel-link group.
PARALLEL_LINK_NAMES = ("Waist", "Left Ankle", "Right Ankle", "Left Wrist", "Right Wrist")
PARALLEL_LINK_JOINTS = ((1, 2), (7, 8), (13, 14), (20, 21), (27, 28))

JOINT_GROUPS: dict[str, list[int]] = {
    "waist": [0, 1, 2],
    "left_leg": [3, 4, 5, 6, 7, 8],
    "right_leg": [9, 10, 11, 12, 13, 14],
    "left_arm": [15, 16, 17, 18, 19, 20, 21],
    "right_arm": [22, 23, 24, 25, 26, 27, 28],
    "neck": [29, 30],
}

# Gains verified with igris_replay. Joints not listed have no default and must be
# given in the demo config before they can be commanded.
DEFAULT_KP: dict[int, float] = {
    0: 50.0, 1: 25.0, 2: 25.0,
    15: 50.0, 16: 50.0, 17: 30.0, 18: 30.0, 19: 15.0, 20: 15.0, 21: 15.0,
    22: 50.0, 23: 50.0, 24: 30.0, 25: 30.0, 26: 15.0, 27: 15.0, 28: 15.0,
}
DEFAULT_KD: dict[int, float] = {
    0: 0.8, 1: 0.8, 2: 0.8,
    15: 0.5, 16: 0.5, 17: 0.15, 18: 0.15, 19: 0.1, 20: 0.1, 21: 0.1,
    22: 0.5, 23: 0.5, 24: 0.15, 25: 0.15, 26: 0.1, 27: 0.1, 28: 0.1,
}

# Hand: rt/handcmd motor ids, in "robot order". Right 11~16, left 21~26.
# (1-DoF gripper: only 11 = right, 21 = left are used.)
HAND_MOTOR_IDS = (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)

# 1-DoF grippers: slot index in the 12-slot HandCmd/HandState array (robot order).
LEFT_GRIPPER_SLOT = 6   # SDK id 21
RIGHT_GRIPPER_SLOT = 0  # SDK id 11


def parse_joint_list(spec) -> list[int]:
    """Accept ints, group names ("left_arm") or "a-b" ranges."""
    if spec is None:
        return []
    if isinstance(spec, (int, str)):
        spec = [spec]
    out: list[int] = []
    for item in spec:
        if isinstance(item, int):
            out.append(item)
        elif item in JOINT_GROUPS:
            out.extend(JOINT_GROUPS[item])
        elif isinstance(item, str) and "-" in item:
            a, b = item.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            raise ValueError(f"Unknown joint spec: {item!r}")
    return out
