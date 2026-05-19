"""
src/conformal/split_conformal.py
--------------------------------
Vanilla split conformal prediction.

Given a calibration set of (forecast, observed) pairs we compute the
absolute-residual quantile and emit symmetric prediction intervals
``[y_hat - q_alpha, y_hat + q_alpha]``.

This is the static baseline used in our experiments; ACI and DtACI
extend it with online adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SplitConformal:
    alpha: float = 0.10

    def __post_init__(self):
        self.q: float = 0.0
        self._fitted = False

    def fit(self, y_hat_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Compute the (1 - alpha) quantile of |y - y_hat| on calibration."""
        if y_hat_cal.shape != y_cal.shape:
            raise ValueError("calibration arrays must have the same shape")
        residuals = np.abs(y_cal - y_hat_cal)
        n = residuals.size
        # Conformal correction: ceil((n+1)(1-alpha))/n quantile.
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / max(n, 1))
        self.q = float(np.quantile(residuals, q_level, method="higher"))
        self._fitted = True

    def interval(self, y_hat: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit() before interval().")
        y_hat = np.asarray(y_hat)
        return y_hat - self.q, y_hat + self.q
