"""Leg retargeting checks (pelvis frame, URDF FK). SMPL frame: x left, y up, z forward.

SMPL: hip flexion (thigh forward) = Rx(-a), knee flexion (shank backward) = Rx(+a),
left abduction = Rz(+a), right abduction = Rz(-a).
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from igris_teleop.retarget import C, IDX, Retargeter, to_robot
from igris_teleop.smpl import local_rotations, pose_from_rotations
from igris_teleop.urdf_fk import UrdfFK

R = lambda axis, deg: Rotation.from_euler(axis, deg, degrees=True).as_matrix()
LEGS = [f"{n}_{s}" for s in "LR" for n in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")]


def q_of(rots):
    return Retargeter()(pose_from_rotations(rots)).q


def test_standing_is_zero():
    q = q_of({})
    for n in LEGS:
        assert q[IDX[n]] == pytest.approx(0.0, abs=1e-9), n


def test_squat_hip_and_knee_90():
    rots = {f"{p}_hip": R("x", -90) for p in ("left", "right")}
    rots.update({f"{p}_knee": R("x", 90) for p in ("left", "right")})
    q = q_of(rots)
    for s in "LR":
        assert q[IDX[f"Hip_Pitch_{s}"]] == pytest.approx(-np.pi / 2, abs=1e-6)
        assert q[IDX[f"Knee_Pitch_{s}"]] == pytest.approx(np.pi / 2, abs=1e-6)
        for n in ("Hip_Roll", "Hip_Yaw", "Ankle_Pitch", "Ankle_Roll"):
            assert q[IDX[f"{n}_{s}"]] == pytest.approx(0.0, abs=1e-6), n


def test_leg_out_to_the_side():
    q = q_of({"left_hip": R("z", 30), "right_hip": R("z", -30)})
    assert q[IDX["Hip_Roll_L"]] == pytest.approx(np.deg2rad(30), abs=1e-6)    # left leg out = +roll
    assert q[IDX["Hip_Roll_R"]] == pytest.approx(-np.deg2rad(30), abs=1e-6)


def test_noisy_forward_knee_bend_does_not_flip_hip_yaw():
    q = q_of({"left_hip": R("x", -20), "left_knee": R("x", -8)})   # hyper-extended (estimation noise)
    assert abs(q[IDX["Hip_Yaw_L"]]) < 1e-6
    assert q[IDX["Knee_Pitch_L"]] == pytest.approx(0.0, abs=1e-9)


def test_random_leg_poses_match_fk():
    """Thigh / shank directions and foot orientation from URDF FK == the person's (pelvis frame)."""
    fk = UrdfFK()
    rng = np.random.default_rng(2)
    down = np.array([0.0, -1.0, 0.0])
    n = 0
    for _ in range(300):
        rots = {}
        for pre, s in (("left", 1), ("right", -1)):
            rots[f"{pre}_hip"] = R("x", -rng.uniform(-20, 100)) @ R("z", s * rng.uniform(-10, 40)) @ R("y", rng.uniform(-30, 30))
            rots[f"{pre}_knee"] = R("x", rng.uniform(15, 110))
            rots[f"{pre}_ankle"] = R("x", rng.uniform(-25, 25)) @ R("z", rng.uniform(-12, 12))
        pose = pose_from_rotations(rots)
        q = Retargeter()(pose).q
        poses = fk.link_poses(q)
        Rp = poses["Link_Waist_Pitch"][0]
        Rl = local_rotations(pose)
        for pre, suf in (("left", "Left"), ("right", "Right")):
            Hh, K, A = (Rl[i] for i in pose_index(pre))
            t_ref, s_ref = C @ (Hh @ down), C @ (Hh @ K @ down)
            t_rb = -(Rp.T @ poses[f"Link_Hip_Yaw_{suf}"][0])[:, 2]
            s_rb = -(Rp.T @ poses[f"Link_Knee_Pitch_{suf}"][0])[:, 2]
            foot = Rp.T @ poses[f"Link_Ankle_Roll_{suf}"][0]
            ang = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
            assert ang(t_ref, t_rb) < 0.5 and ang(s_ref, s_rb) < 0.5, (pre, ang(t_ref, t_rb), ang(s_ref, s_rb))
            assert np.degrees(Rotation.from_matrix(to_robot(Hh @ K @ A).T @ foot).magnitude()) < 0.5
            n += 1
    assert n == 600


def pose_index(pre):
    from igris_teleop.smpl import J
    return J[f"{pre}_hip"], J[f"{pre}_knee"], J[f"{pre}_ankle"]
