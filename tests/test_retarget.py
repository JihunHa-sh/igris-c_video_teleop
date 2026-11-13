"""Retargeter checks on synthetic SMPL poses (no camera, no robot).

1. Named poses with known IGRIS answers (T-pose, arms down, arms forward, elbow 90 deg, torso lean/twist).
2. Random arm poses: the robot's upper-arm / forearm directions computed by URDF forward
   kinematics must match the person's bone directions (validates every sign and axis).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from igris_teleop.retarget import IDX, Retargeter, to_robot
from igris_teleop.smpl import J, chain, local_rotations, pose_from_rotations
from igris_teleop.urdf_fk import UrdfFK

R = lambda axis, deg: Rotation.from_euler(axis, deg, degrees=True).as_matrix()  # SMPL frame: x left, y up, z fwd

ARMS_DOWN = {"left_shoulder": R("z", -90), "right_shoulder": R("z", 90)}


def q_of(rots):
    return Retargeter()(pose_from_rotations(rots)).q


def test_tpose_is_shoulder_roll_90():
    q = q_of({})
    assert q[IDX["Shoulder_Roll_L"]] == pytest.approx(np.pi / 2, abs=1e-6)
    assert q[IDX["Shoulder_Roll_R"]] == pytest.approx(-np.pi / 2, abs=1e-6)
    assert q[IDX["Elbow_Pitch_L"]] == pytest.approx(0, abs=1e-6)


def test_arms_down_is_zero():
    q = q_of(ARMS_DOWN)
    for n in ("Shoulder_Pitch", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Yaw", "Wrist_Roll", "Wrist_Pitch"):
        for s in ("_L", "_R"):
            assert q[IDX[n + s]] == pytest.approx(0, abs=1e-6), n + s
    for n in ("Waist_Yaw", "Waist_Roll", "Waist_Pitch"):
        assert q[IDX[n]] == pytest.approx(0, abs=1e-6)


def test_arms_forward_is_shoulder_pitch_minus_90():
    # left arm from +x (left) to +z (forward): rotate about y by -90
    q = q_of({"left_shoulder": R("y", -90), "right_shoulder": R("y", 90)})
    assert q[IDX["Shoulder_Pitch_L"]] == pytest.approx(-np.pi / 2, abs=1e-6)
    assert q[IDX["Shoulder_Pitch_R"]] == pytest.approx(-np.pi / 2, abs=1e-6)
    assert abs(q[IDX["Shoulder_Roll_L"]]) < 1e-6 and abs(q[IDX["Shoulder_Roll_R"]]) < 1e-6


def test_elbow_90_forward_from_arms_down():
    # arms down, forearm forward: elbow local rotation brings forearm toward +z
    rots = dict(ARMS_DOWN)
    rots["left_elbow"] = R("y", -90)   # in the upper-arm frame (x = along arm in T-pose)
    rots["right_elbow"] = R("y", 90)
    q = q_of(rots)
    assert q[IDX["Elbow_Pitch_L"]] == pytest.approx(-np.pi / 2, abs=1e-6)
    assert q[IDX["Elbow_Pitch_R"]] == pytest.approx(-np.pi / 2, abs=1e-6)
    assert abs(q[IDX["Shoulder_Yaw_L"]]) < 1e-6 and abs(q[IDX["Shoulder_Yaw_R"]]) < 1e-6


def test_torso_lean_forward_and_twist():
    # lean forward 20 deg: torso z-up tilts toward +z_smpl -> rotation about +x_smpl (left axis)
    lean = q_of({**ARMS_DOWN, "spine1": R("x", 20)})
    # robot: torso pitched forward = Ry(+20deg) in pelvis frame; chain torso->pelvis -> waist pitch = -20 deg
    assert lean[IDX["Waist_Pitch"]] == pytest.approx(-np.deg2rad(20), abs=1e-6)
    twist = q_of({**ARMS_DOWN, "spine2": R("y", 30)})  # torso turns to the person's left
    assert twist[IDX["Waist_Yaw"]] == pytest.approx(-np.deg2rad(30), abs=1e-6)


def _smpl_dirs(pose, side):
    s = 1.0 if side == "L" else -1.0
    pre = "left" if side == "L" else "right"
    Rl = local_rotations(pose)
    U = to_robot(chain(Rl, f"{pre}_collar", f"{pre}_shoulder"))
    E = to_robot(Rl[J[f"{pre}_elbow"]])
    rest = np.array([0, s, 0.0])
    return U @ rest, U @ E @ rest


def test_random_arm_poses_match_fk_directions():
    """Bone directions from URDF FK on the retargeted angles == the person's bone directions."""
    fk = UrdfFK()
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(400):
        rots = {}
        for side, pre, s in (("L", "left", 1), ("R", "right", -1)):
            # upper arm anywhere in a reachable cone (down/forward/side), elbow flexed 20..120 deg
            sh = Rotation.from_euler("zy", [rng.uniform(-90, 0) * s, rng.uniform(-80, 30) * s], degrees=True)
            el = Rotation.from_euler("y", -rng.uniform(20, 120) * s, degrees=True)
            rots[f"{pre}_shoulder"] = sh.as_matrix()
            rots[f"{pre}_elbow"] = el.as_matrix()
        pose = pose_from_rotations(rots)
        q = Retargeter()(pose).q
        poses = fk.link_poses(q)
        for side, suf in (("L", "Left"), ("R", "Right")):
            u_ref, f_ref = _smpl_dirs(pose, side)
            sh = poses[f"Link_Shoulder_Pitch_{suf}"][1]
            el = poses[f"Link_Elbow_Pitch_{suf}"][1]
            wr = poses[f"Link_Wrist_Roll_{suf}"][1]
            u_rb = (el - sh) / np.linalg.norm(el - sh)
            f_rb = (wr - el) / np.linalg.norm(wr - el)
            # URDF links have small lateral offsets (1 cm elbow, 0.5 cm wrist): allow ~5 deg
            within = (np.degrees(np.arccos(np.clip(u_ref @ u_rb, -1, 1))) < 5
                      and np.degrees(np.arccos(np.clip(f_ref @ f_rb, -1, 1))) < 6)
            lim_ok = all(abs(q[IDX[n + ("_L" if side == "L" else "_R")]]) <= np.pi / 2 + 1e-9
                         for n in ("Shoulder_Yaw",))
            if lim_ok:  # skip poses needing > +-90 deg shoulder yaw (outside the robot's range)
                assert within, (side, np.degrees(np.arccos(u_ref @ u_rb)), np.degrees(np.arccos(f_ref @ f_rb)))
                n_ok += 1
    assert n_ok > 500


def test_random_wrist_orientation_matches_fk():
    """With the arm solved, the robot hand frame (URDF FK) equals the person's hand frame."""
    from igris_teleop.retarget import rot_x
    fk = UrdfFK()
    rng = np.random.default_rng(1)
    for _ in range(200):
        rots = {}
        for pre, s in (("left", 1), ("right", -1)):
            rots[f"{pre}_shoulder"] = Rotation.from_euler("zy", [rng.uniform(-80, -10) * s, rng.uniform(-60, 0) * s], degrees=True).as_matrix()
            rots[f"{pre}_elbow"] = Rotation.from_euler("y", -rng.uniform(30, 100) * s, degrees=True).as_matrix()
            # small wrist rotations so the robot wrist limits are not hit
            rots[f"{pre}_wrist"] = Rotation.from_rotvec(rng.uniform(-0.35, 0.35, 3)).as_matrix()
        pose = pose_from_rotations(rots)
        q = Retargeter()(pose).q
        poses = fk.link_poses(q)
        Rl = local_rotations(pose)
        for pre, s, suf in (("left", 1, "Left"), ("right", -1, "Right")):
            H = to_robot(chain(Rl, f"{pre}_collar", f"{pre}_shoulder", f"{pre}_elbow", f"{pre}_wrist")) @ rot_x(s * np.pi / 2)
            R_rb = poses[f"Link_Wrist_Pitch_{suf}"][0]
            ang = np.degrees(Rotation.from_matrix(H.T @ R_rb).magnitude())
            assert ang < 0.5, (pre, ang)
