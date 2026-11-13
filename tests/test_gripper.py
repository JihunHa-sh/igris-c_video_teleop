import numpy as np

from igris_teleop.gripper import HandSwitch


def run(sw, seq, dt=0.033):
    out = []
    for k, c in enumerate(seq):
        sw.update(c, k * dt)
        out.append(sw.closed)
    return out


def test_fist_closes_after_hold_and_open_hand_opens():
    sw = HandSwitch(close_above=0.6, open_below=0.35, hold_s=0.15)
    states = run(sw, [0.1] * 10 + [0.9] * 10 + [0.1] * 10)
    assert not any(states[:10])
    assert not states[11] and states[19]          # needs ~0.15 s of fist
    assert not states[-1]                         # opened again


def test_half_closed_and_noise_do_not_chatter():
    sw = HandSwitch(close_above=0.6, open_below=0.35, hold_s=0.15)
    rng = np.random.default_rng(0)
    half = list(0.47 + 0.1 * rng.standard_normal(200))   # hovering between the thresholds
    states = run(sw, half)
    assert sum(np.diff(states) != 0) == 0
    spikes = [0.1] * 20 + [0.95, 0.1] * 20                  # single-frame false fists
    assert not any(run(HandSwitch(), spikes))


def test_nan_keeps_state():
    sw = HandSwitch(hold_s=0.0)
    run(sw, [0.9, 0.9])
    assert sw.closed
    run(sw, [float("nan")] * 20)
    assert sw.closed
