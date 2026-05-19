"""
src/conformal/aci.py
--------------------
Adaptive Conformal Inference (Gibbs & Candes, 2021).

Online update rule
------------------
At each step we observe (y_hat_t, y_t). We maintain an effective
miscoverage rate alpha_t and emit the interval

    [y_hat_t - q_{1-alpha_t}, y_hat_t + q_{1-alpha_t}]

where the quantile is taken over the (rolling) residual buffer.

After observing y_t we compute the miscoverage indicator

    err_t = 1 [ y_t not in interval ]

and update

    alpha_{t+1} = clip(alpha_t + eta * (alpha_target - err_t),
                      alpha_min, alpha_max)

so that when we *over*-cover (err_t=0) alpha grows (narrower interval)
and when we *under*-cover alpha shrinks (wider interval), driving the
long-run empirical miscoverage toward alpha_target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class ACI:
    alpha_target: float = 0.10
    eta: float = 0.05
    alpha_min: float = 0.005
    alpha_max: float = 0.30
    buffer_size: int = 200

    def __post_init__(self):
        self.alpha_t: float = float(self.alpha_target)
        self.residuals: List[float] = []
        self.coverage_log: List[int] = []
        self.alpha_log: List[float] = []
        self.width_log: List[float] = []

    # ------------------------------------------------------------------
    def warm_up(self, y_hat_init: np.ndarray, y_init: np.ndarray) -> None:
        """Seed the residual buffer with calibration data."""
        resid = np.abs(y_init - y_hat_init).tolist()
        self.residuals = resid[-self.buffer_size:]

    # ------------------------------------------------------------------
    def interval(self, y_hat: float) -> Tuple[float, float]:
        if not self.residuals:
            return float(y_hat), float(y_hat)
        # Use higher method to be conservative.
        q_level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        q = float(np.quantile(self.residuals, q_level, method="higher"))
        lo = float(y_hat) - q
        hi = float(y_hat) + q
        self.width_log.append(hi - lo)
        return lo, hi

    # ------------------------------------------------------------------
    def update(self, y_hat: float, y_true: float,
               lo: float, hi: float) -> None:
        """Online update after the truth is revealed."""
        covered = int(lo <= y_true <= hi)
        err = 1 - covered
        # Step the effective alpha.
        self.alpha_t = float(np.clip(
            self.alpha_t + self.eta * (self.alpha_target - err),
            self.alpha_min, self.alpha_max,
        ))
        # Append residual and trim.
        self.residuals.append(abs(y_true - y_hat))
        if len(self.residuals) > self.buffer_size:
            self.residuals = self.residuals[-self.buffer_size:]
        self.coverage_log.append(covered)
        self.alpha_log.append(self.alpha_t)

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        return {
            "empirical_coverage": (np.mean(self.coverage_log)
                                   if self.coverage_log else float("nan")),
            "avg_width": (np.mean(self.width_log)
                          if self.width_log else float("nan")),
            "alpha_final": self.alpha_t,
        }
