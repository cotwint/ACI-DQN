"""
src/rl/augmenters.py
--------------------
State augmenters used by ACI-DQN and DtACI-DQN.

An augmenter exposes the protocol::

    reset(env, day_index)
    augment(state, env) -> np.ndarray
    shield(action, state, env) -> int
    on_step(env, info, action_pre, action_post)
    metrics() -> dict

``IdentityAugmenter`` lives in ``train_dqn.py`` (no extra inputs, no shield).
``ConformalAugmenter`` adds DtACI / ACI intervals to the state and (optionally)
runs them through ``DtACIActionShield``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..conformal.aci import ACI
from ..conformal.dtaci import DtACI
from ..conformal.forecaster import build_forecaster
from ..datacenter_env import DataCenterEnv, action_to_n, n_to_action
from ..safe_layer.dtaci_action_shield import DtACIActionShield


# ---------------------------------------------------------------------------
# Per-priority forecaster + online conformal learner
# ---------------------------------------------------------------------------

class PerPriorityConformalForecaster:
    """Maintain K forecasters + K conformal learners (one per priority).

    For simplicity we use rolling-mean point forecasts on the realised
    arrival counts; each step's residual feeds the online learner.
    All updates use only past data, so no test-set leakage occurs.
    """

    def __init__(self, cfg: Dict, learner: str = "dtaci"):
        self.cfg = cfg
        self.K = int(cfg["qos"]["K"])
        self.H = int(cfg["conformal"]["horizon"])
        self.window = int(cfg["conformal"]["rolling_window"])
        kind = cfg["conformal"]["forecaster"]
        self.forecasters = [
            build_forecaster(kind, window=self.window, horizon=self.H)
            for _ in range(self.K)
        ]
        self.learner_kind = learner
        self._build_learners()
        self.history: List[np.ndarray] = []   # per-slot (K,) realised arrivals

    # ------------------------------------------------------------------
    def _build_learners(self):
        c = self.cfg["conformal"]
        if self.learner_kind == "aci":
            self.learners = [ACI(alpha_target=c["alpha"], eta=c["aci_eta"],
                                  alpha_min=c["alpha_min"],
                                  alpha_max=c["alpha_max"])
                             for _ in range(self.K)]
        elif self.learner_kind == "dtaci":
            self.learners = [DtACI(alpha_target=c["alpha"],
                                    etas=c["dtaci_etas"],
                                    alpha_min=c["alpha_min"],
                                    alpha_max=c["alpha_max"],
                                    sigma=c["dtaci_sigma"],
                                    meta_lr=c["dtaci_meta_lr"])
                             for _ in range(self.K)]
        else:
            raise ValueError(f"Unknown conformal learner {self.learner_kind}")

    # ------------------------------------------------------------------
    def reset_episode(self):
        self.history = []

    # ------------------------------------------------------------------
    def warm_up_from_calibration(self,
                                 past_lam: np.ndarray,
                                 past_real: np.ndarray) -> None:
        """Seed the conformal residual buffers with calibration data."""
        for k in range(self.K):
            self.learners[k].warm_up(past_lam[:, k], past_real[:, k])

    # ------------------------------------------------------------------
    def intervals_h_steps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (K, H) lower and upper bounds for the next H steps."""
        K, H = self.K, self.H
        lo = np.zeros((K, H))
        hi = np.zeros((K, H))
        for k in range(K):
            hist_k = np.array([h[k] for h in self.history], dtype=np.float64)
            y_hat = self.forecasters[k].predict(hist_k, h=1)
            l, u = self.learners[k].interval(y_hat)
            lo[k, :] = max(0.0, l)
            hi[k, :] = max(0.0, u)
        return lo, hi

    # ------------------------------------------------------------------
    def update_after_step(self, info: dict) -> None:
        arrivals = np.asarray(info["arrivals"], dtype=np.float64)
        for k in range(self.K):
            hist_k = np.array([h[k] for h in self.history], dtype=np.float64)
            y_hat = self.forecasters[k].predict(hist_k, h=1)
            lo, hi = self.learners[k].interval(y_hat)
            self.learners[k].update(y_hat, arrivals[k], lo, hi)
        self.history.append(arrivals.copy())

    # ------------------------------------------------------------------
    def metrics(self) -> Dict:
        return {f"P{k+1}": self.learners[k].metrics()
                for k in range(self.K)}


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------

class ConformalAugmenter:
    """Adds conformal intervals to the state; optionally shields the action.

    Parameters
    ----------
    cfg : full config dict
    learner : "aci" or "dtaci"
    use_shield : whether to apply the DtACI action shield
    """

    def __init__(self, cfg: Dict, learner: str, use_shield: bool):
        self.cfg = cfg
        self.learner_kind = learner
        self.use_shield = bool(use_shield)
        self.cp = PerPriorityConformalForecaster(cfg, learner=learner)
        self._shield_obj = (DtACIActionShield(cfg) if use_shield else None)
        self.name = learner + ("_shielded" if use_shield else "")
        self._cached_hi: np.ndarray | None = None
        # Normalisation divisor for conformal features (prevent huge inputs).
        self._lambda_scale = float(cfg["conformal"].get("lambda_scale", 20.0))

    # ------------------------------------------------------------------
    def reset(self, env: DataCenterEnv, day_index: int) -> None:
        self.cp.reset_episode()
        if self._shield_obj is not None:
            self._shield_obj.reset_log()
        self._cached_hi = None

    # ------------------------------------------------------------------
    def augment(self, state: np.ndarray, env: DataCenterEnv) -> np.ndarray:
        lo, hi = self.cp.intervals_h_steps()
        # Compact 2K-dim extra feature: per-priority mean (upper, lower) bound,
        # normalised by lambda_scale to avoid saturating the Q-network.
        extra = np.concatenate([
            hi.mean(axis=1) / max(0.01, self._lambda_scale),
            lo.mean(axis=1) / max(0.01, self._lambda_scale),
        ], axis=0).astype(np.float32)
        self._cached_hi = hi
        return np.concatenate([state, extra]).astype(np.float32)

    # ------------------------------------------------------------------
    def shield(self, n_servers: int, state, env: DataCenterEnv) -> int:
        # n_servers is already a server count (converted in rollout_episode).
        # Shield returns the (possibly adjusted) server count.
        if self._shield_obj is None or self._cached_hi is None:
            return int(n_servers)
        q_lengths = np.array([len(q) for q in env.queues], dtype=np.float64)
        n_safe = self._shield_obj.filter(int(n_servers), q_lengths, self._cached_hi)
        return int(n_safe)

    # ------------------------------------------------------------------
    def on_step(self, env: DataCenterEnv, info: dict,
                action_pre: int, action_post: int) -> None:
        self.cp.update_after_step(info)

    # ------------------------------------------------------------------
    def metrics(self) -> Dict:
        out = {"conformal": self.cp.metrics()}
        if self._shield_obj is not None:
            out["shield_mod_rate"] = self._shield_obj.log.mod_rate
            out["shield_avg_mod_size"] = self._shield_obj.log.avg_mod_size
            out["shield_n_steps"] = self._shield_obj.log.n_steps
        return out
