"""WalkFollower: person offset from the home spot -> walk velocity."""

import math

import numpy as np

from igris_teleop.walk import WalkFollower

CFG = dict(deadzone_fwd_m=0.2, deadzone_side_m=0.2, deadzone_turn_deg=20, hold_s=0.3, smooth=1.0,
           max_vx=0.2, max_vy=0.1, max_wz=0.3, acc_lin=100.0, acc_yaw=100.0)


def run(w, p, h, secs, enabled=True, dt=0.02, t0=0.0):
    v = None
    for k in range(int(secs / dt)):
        v = w.update(p, h, t0 + k * dt, dt, enabled)
    return v


def make(home_p=(3.0, 0.0), home_h=math.pi):
    w = WalkFollower(CFG)
    assert w.set_home(np.array([*home_p, -0.8]), home_h)
    return w


def test_standing_at_home_is_zero():
    w = make()
    assert np.allclose(run(w, [3.0, 0.05, -0.8], math.pi + 0.1, 1.0), 0)


def test_step_toward_camera_is_forward():
    w = make()                                         # facing the camera (heading pi): forward = -e1
    v = run(w, [2.5, 0.0, -0.8], math.pi, 1.0)
    assert v[0] > 0.19 and abs(v[1]) < 1e-9 and abs(v[2]) < 1e-9


def test_step_to_own_left_is_plus_y():
    w = make()                                         # facing the camera, own left = camera's -e2 side
    v = run(w, [3.0, -0.5, -0.8], math.pi, 1.0)
    assert v[1] > 0.09 and abs(v[0]) < 1e-9


def test_turn_left_is_plus_wz():
    w = make()
    v = run(w, [3.0, 0.0, -0.8], math.pi + math.radians(40), 1.0)
    assert v[2] > 0.29


def test_hold_time_and_disable_and_missing():
    w = make()
    assert np.allclose(run(w, [2.5, 0, -0.8], math.pi, 0.2), 0)             # < hold_s: not yet
    assert run(w, [2.5, 0, -0.8], math.pi, 0.5, t0=0.2)[0] > 0
    assert np.allclose(run(w, [2.5, 0, -0.8], math.pi, 0.5, enabled=False), 0)
    assert np.allclose(run(w, [np.nan, np.nan, np.nan], math.pi, 0.5), 0)
    assert np.allclose(run(w, [2.5, 0, -0.8], math.pi + math.radians(120), 0.5), 0)  # implausible turn


def test_no_home_never_walks():
    w = WalkFollower(CFG)
    assert not w.set_home(None, None)
    assert np.allclose(run(w, [2.0, 0, -0.8], math.pi, 1.0), 0)
