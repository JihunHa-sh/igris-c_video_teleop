"""SMPL skeleton facts used by the retargeter (no SMPL model file needed).

Fast SAM 3D Body publishes, per frame (SONIC protocol v2):
  body_quat   (4,)     root orientation in the gravity-aligned world, WXYZ
  smpl_joints (24, 3)  joint positions, root-relative, in the root-local frame
  smpl_pose   (21, 3)  axis-angle LOCAL rotations of joints 1..21 (standard SMPL body_pose)

Only `smpl_pose` drives the robot: local rotations are independent of where the
person stands or which way the camera looks.

SMPL canonical frame: x = the person's left, y = up, z = forward (facing direction).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
J = {n: i for i, n in enumerate(JOINT_NAMES)}

# Bones drawn by the viewers.
BONES = tuple((PARENTS[i], i) for i in range(1, 22))


def local_rotations(smpl_pose: np.ndarray) -> np.ndarray:
    """(21, 3) axis-angle -> (24, 3, 3) local rotation matrices (pelvis/hands = identity)."""
    pose = np.asarray(smpl_pose, dtype=np.float64).reshape(21, 3)
    R = np.tile(np.eye(3), (24, 1, 1))
    R[1:22] = Rotation.from_rotvec(pose).as_matrix()
    return R


def chain(R_local: np.ndarray, *names: str) -> np.ndarray:
    """Product of local rotations along consecutive joints, e.g. chain(R, 'left_collar', 'left_shoulder')."""
    out = np.eye(3)
    for n in names:
        out = out @ R_local[J[n]]
    return out


def pose_from_rotations(rots: dict[str, np.ndarray]) -> np.ndarray:
    """Build a (21, 3) smpl_pose from {joint name: 3x3 local rotation}; others = identity."""
    pose = np.zeros((21, 3))
    for n, R in rots.items():
        pose[J[n] - 1] = Rotation.from_matrix(R).as_rotvec()
    return pose


# Approximate rest-pose joint positions (m, SMPL canonical frame, pelvis at origin).
# Hand-written round numbers for drawing / synthetic tests only - not SMPL model data.
REST_JOINTS = np.array([
    [0.00, 0.00, 0.00], [0.06, -0.09, 0.00], [-0.06, -0.09, 0.00], [0.00, 0.11, -0.02],
    [0.10, -0.47, 0.00], [-0.10, -0.47, 0.00], [0.00, 0.25, 0.00], [0.09, -0.87, -0.04],
    [-0.09, -0.87, -0.04], [0.00, 0.30, 0.02], [0.12, -0.92, 0.08], [-0.12, -0.92, 0.08],
    [0.00, 0.51, -0.01], [0.08, 0.42, 0.00], [-0.08, 0.42, 0.00], [0.00, 0.60, 0.04],
    [0.17, 0.45, -0.01], [-0.17, 0.45, -0.01], [0.43, 0.43, -0.03], [-0.43, 0.43, -0.03],
    [0.68, 0.44, -0.03], [-0.68, 0.44, -0.03], [0.76, 0.43, -0.04], [-0.76, 0.43, -0.04],
])


def forward_kinematics(smpl_pose: np.ndarray, rest: np.ndarray = REST_JOINTS):
    """Joint positions (24, 3) and global rotations (24, 3, 3) relative to the pelvis (canonical frame)."""
    Rl = local_rotations(smpl_pose)
    Rg = np.zeros_like(Rl)
    p = np.zeros((24, 3))
    for j in range(24):
        par = PARENTS[j]
        if par < 0:
            Rg[j] = Rl[j]
            p[j] = rest[j]
        else:
            Rg[j] = Rg[par] @ Rl[j]
            p[j] = p[par] + Rg[par] @ (rest[j] - rest[par])
    return p, Rg
