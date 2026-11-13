"""1-DoF gripper commands from the person's hand: fist -> CLOSE, open hand -> OPEN.

The GPU-side publisher sends `hand_closure` = [left, right] in 0 (open) .. 1 (fist).
Each gripper is a two-state switch (like a button command), not a proportional
follower:
  OPEN  -> CLOSED  when closure stays above `close_above` for `hold_s`
  CLOSED -> OPEN   when closure stays below `open_below`  for `hold_s`
The gap between the two thresholds (hysteresis) and the hold time stop the gripper
from chattering when the estimate is noisy or the hand is half closed.
Missing / NaN closure keeps the current state.
"""

from __future__ import annotations

import math


class HandSwitch:
    def __init__(self, close_above=0.6, open_below=0.35, hold_s=0.15, closed=False):
        if open_below >= close_above:
            raise ValueError("open_below must be < close_above")
        self.close_above, self.open_below, self.hold_s = close_above, open_below, hold_s
        self.closed = closed
        self._since: float | None = None   # time the opposite condition started

    def reset_timer(self) -> None:
        self._since = None

    def update(self, closure: float | None, t: float) -> bool:
        """Feed one closure sample; returns True when the state changed."""
        if closure is None or math.isnan(closure):
            self._since = None
            return False
        want_flip = closure < self.open_below if self.closed else closure > self.close_above
        if not want_flip:
            self._since = None
            return False
        if self._since is None:
            self._since = t
        if t - self._since >= self.hold_s:
            self.closed = not self.closed
            self._since = None
            return True
        return False
