"""Hand closure (0 = open, 1 = fist) from SAM 3D Body MHR70 3D keypoints.

Per finger (index, middle, ring, pinky) the straightness is
    |tip - wrist| / (|tip-j1| + |j1-j2| + |j2-j3| + |j3-wrist|)
which is ~1 for a straight finger and drops as the finger curls. It is a ratio of
lengths, so it does not depend on hand size, camera distance or orientation.
The thumb is ignored (it moves a lot without grasping).
"""

from __future__ import annotations

import numpy as np

# MHR70 layout (sam_3d_body/metadata/mhr70.py): per hand 5 fingers x (tip, first, second, third joint) + wrist
RIGHT_BASE, RIGHT_WRIST = 21, 41
LEFT_BASE, LEFT_WRIST = 42, 62
FINGERS = (1, 2, 3, 4)            # index, middle, ring, pinky (0 = thumb)

R_OPEN = 0.90                     # mean straightness of a relaxed open hand -> closure 0
R_FIST = 0.55                     # mean straightness of a fist                -> closure 1


def straightness(kp: np.ndarray, base: int, wrist: int) -> float:
    kp = np.asarray(kp, dtype=np.float64)
    w = kp[wrist]
    vals = []
    for f in FINGERS:
        tip, j1, j2, j3 = (kp[base + 4 * f + k] for k in range(4))
        path = (np.linalg.norm(tip - j1) + np.linalg.norm(j1 - j2) + np.linalg.norm(j2 - j3)
                + np.linalg.norm(j3 - w))
        if path > 1e-6:
            vals.append(np.linalg.norm(tip - w) / path)
    return float(np.mean(vals)) if vals else float("nan")


def closure_from_straightness(r: float, r_open: float = R_OPEN, r_fist: float = R_FIST) -> float:
    return float(np.clip((r_open - r) / (r_open - r_fist), 0.0, 1.0))


def hand_state(kp70: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(70, 3) keypoints -> (closure[left, right], straightness[left, right])."""
    r = np.array([straightness(kp70, LEFT_BASE, LEFT_WRIST), straightness(kp70, RIGHT_BASE, RIGHT_WRIST)])
    c = np.array([closure_from_straightness(v) for v in r])
    return c, r
