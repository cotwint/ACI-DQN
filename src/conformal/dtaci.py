"""
src/conformal/dtaci.py
----------------------
Dynamically-tuned Adaptive Conformal Inference (DtACI).

Reference
---------
Gibbs & Candes (2022), "Conformal Inference for Online Prediction with
Arbitrary Distribution Shifts". The original paper proposes to combine
several ACI learners that differ in their step size ``eta`` and to
update their weights with an exponential-weight no-regret algorithm so
that the *effective* learning rate adapts to the local non-stationarity
of the data.

Algorithm sketch
----------------
For each expert m we keep its own running alpha^{(m)}_t and at every
step we compute the candidate interval and update alpha^{(m)} with the
ACI rule. We then form a *mixture* alpha_t = sum_m w^{(m)}_t alpha^{(m)}_t
and use it to emit the final interval.

The mixture weights w^{(m)} are updated by exponential weights on the
pinball loss

    L_t^{(m)} = alpha_target * max(0, err_t^{(m)} - alpha_target)
              + (1 - alpha_target) * max(0, alpha_target - err_t^{(m)})

Concretely (using \\propto for "proportional to" — kept as raw string):

    w_{t+1}^{(m)} = (1/Z_t) * w_t^{(m)} * exp(- sigma * L_t^{(m)})

followed by mild smoothing w <- (1 - meta_lr) w + meta_lr / M
to encourage exploration when one expert becomes dominant. The
selected/adaptive learning rate at time t is reported as the weighted
mean of the experts' etas. This serves as the "dynamic tuning" output
demanded by the spec.

The DtACI loop is fully online: at each time step we
  1) compute mixture alpha and emit interval;
  2) observe y_true, evaluate per-expert errors;
  3) update each expert's alpha via ACI step;
  4) update mixture weights via exponential weighting + smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class _Expert:
    eta: float
    alpha_target: float
    alpha_min: float
    alpha_max: float
    buffer_size: int
    alpha_t: float = 0.0
    residuals: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.alpha_t = float(self.alpha_target)

    def warm_up(self, init_residuals: np.ndarray) -> None:
        self.residuals = list(map(float, init_residuals[-self.buffer_size:]))

    def quantile(self) -> float:
        if not self.residuals:
            return 0.0
        q_level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        return float(np.quantile(self.residuals, q_level, method="higher"))

    def step(self, y_true: float, y_hat: float,
             covered: int) -> None:
        err = 1 - covered
        self.alpha_t = float(np.clip(
            self.alpha_t + self.eta * (self.alpha_target - err),
            self.alpha_min, self.alpha_max,
        ))
        self.residuals.append(abs(y_true - y_hat))
        if len(self.residuals) > self.buffer_size:
            self.residuals = self.residuals[-self.buffer_size:]


class DtACI:
    """Dynamically-tuned ACI with expert weighting.

    Attributes
    ----------
    experts : list of ACI experts with different eta values.
    weights : current mixture weights over experts.
    """

    def __init__(self,
                 alpha_target: float = 0.10,
                 etas: List[float] | None = None,
                 alpha_min: float = 0.005,
                 alpha_max: float = 0.30,
                 sigma: float = 0.10,
                 meta_lr: float = 0.10,
                 buffer_size: int = 200):
        if etas is None:
            etas = [0.005, 0.02, 0.05, 0.1, 0.25]
        self.alpha_target = float(alpha_target)
        self.sigma = float(sigma)
        self.meta_lr = float(meta_lr)
        self.experts = [_Expert(eta=float(e),
                                alpha_target=alpha_target,
                                alpha_min=alpha_min,
                                alpha_max=alpha_max,
                                buffer_size=buffer_size)
                        for e in etas]
        self.M = len(self.experts)
        self.weights = np.ones(self.M) / self.M

        # Logging
        self.coverage_log: List[int] = []
        self.alpha_log: List[float] = []
        self.width_log: List[float] = []
        self.weights_log: List[np.ndarray] = []
        self.eta_eff_log: List[float] = []

    # ------------------------------------------------------------------
    def warm_up(self, y_hat_init: np.ndarray, y_init: np.ndarray) -> None:
        resid = np.abs(y_init - y_hat_init)
        for e in self.experts:
            e.warm_up(resid)

    # ------------------------------------------------------------------
    def interval(self, y_hat: float) -> Tuple[float, float]:
        # Per-expert quantile then weighted sum (treat half-widths as
        # mixture weights -> robust scalar combination).
        qs = np.array([e.quantile() for e in self.experts])
        q_mix = float(np.dot(self.weights, qs))
        lo, hi = float(y_hat) - q_mix, float(y_hat) + q_mix
        self.width_log.append(hi - lo)
        # Report adaptive alpha and eta.
        alphas = np.array([e.alpha_t for e in self.experts])
        etas = np.array([e.eta for e in self.experts])
        self.alpha_log.append(float(np.dot(self.weights, alphas)))
        self.eta_eff_log.append(float(np.dot(self.weights, etas)))
        return lo, hi

    # ------------------------------------------------------------------
    def update(self, y_hat: float, y_true: float,
               lo: float, hi: float) -> None:
        # Per-expert coverage indicator
        cov_mix = int(lo <= y_true <= hi)
        self.coverage_log.append(cov_mix)

        # For each expert, evaluate its OWN interval using its own
        # quantile and update its alpha + residual buffer.
        per_err = []
        for e in self.experts:
            q_e = e.quantile()
            lo_e = float(y_hat) - q_e
            hi_e = float(y_hat) + q_e
            covered_e = int(lo_e <= y_true <= hi_e)
            per_err.append(1 - covered_e)
            e.step(y_true=y_true, y_hat=y_hat, covered=covered_e)

        # Pinball-style loss on the coverage indicator per expert.
        err_arr = np.asarray(per_err, dtype=np.float64)
        loss = (self.alpha_target * np.maximum(0.0, err_arr - self.alpha_target)
                + (1.0 - self.alpha_target)
                * np.maximum(0.0, self.alpha_target - err_arr))

        # Exponential-weight update with numerical stabilisation.
        log_w = np.log(self.weights + 1e-12) - self.sigma * loss
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        # Mild smoothing for exploration / forgetting.
        w = (1.0 - self.meta_lr) * w + self.meta_lr / self.M
        self.weights = w
        self.weights_log.append(self.weights.copy())

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        return {
            "empirical_coverage": (np.mean(self.coverage_log)
                                   if self.coverage_log else float("nan")),
            "avg_width": (np.mean(self.width_log)
                          if self.width_log else float("nan")),
            "weights_final": self.weights.tolist(),
            "eta_eff_final": (self.eta_eff_log[-1]
                              if self.eta_eff_log else float("nan")),
        }
