"""One Euro filter (Casiez et al. 2012), vectorised, NaN-aware."""

from __future__ import annotations

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff=1.5, beta=0.4, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x = None
        self.dx = None
        self.t = None

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.t is None or t <= self.t:
            if self.x is None:
                self.x = x.copy()
                self.dx = np.zeros_like(x)
                self.t = t
            return self.x.copy()
        dt = t - self.t
        self.t = t
        valid = ~np.isnan(x) & ~np.isnan(self.x)
        dx = np.where(valid, (x - self.x) / dt, 0.0)
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = _alpha(cutoff, dt)
        self.x = np.where(valid, self.x + a * (x - self.x), x)
        return self.x.copy()
