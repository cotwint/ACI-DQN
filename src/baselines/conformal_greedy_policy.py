"""Conformal-Greedy: uses ACI online-adaptive upper bound of next H steps
for conservative capacity planning.

Same rolling-mean forecaster + ACI learner as ACI-DQN, but as a rule-based
policy (no RL). Requires calibration warm-up before use.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..datacenter_env import n_to_action
from ..rl.augmenters import PerPriorityConformalForecaster


class ConformalGreedyPolicy:
    name = "conformal_greedy"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.Nmin = int(cfg["server"]["Nmin"])
        self.Nmax = int(cfg["server"]["Nmax"])
        self.cap = float(cfg["server"]["cap_per_server"])
        self.target_util = float(cfg["server"]["target_util"])
        self.ramp = int(cfg["server"]["ramp_limit"])
        self.deadlines = np.asarray(cfg["qos"]["deadline_slots"], dtype=int)
        self.K = int(cfg["qos"]["K"])
        self.H = int(cfg["conformal"]["horizon"])
        self.service_mean = np.asarray(cfg["qos"]["service_mean"], dtype=np.float64)

        # Same forecaster + ACI learner as ACI-DQN
        self.cp = PerPriorityConformalForecaster(cfg, learner="aci")

    # ------------------------------------------------------------------
    def reset(self):
        self.cp.reset_episode()

    # ------------------------------------------------------------------
    def warm_up_from_calibration(self, y_hat_cal, y_cal) -> None:
        """Seed conformal residual buffers from calibration data."""
        self.cp.warm_up_from_calibration(y_hat_cal, y_cal)

    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, env) -> int:
        # ACI upper bounds for next H steps
        lo, hi = self.cp.intervals_h_steps()   # (K, H)

        current_work = np.array(
            [sum(task.remaining for task in q) for q in env.queues],
            dtype=np.float64,
        )
        near_work = self._near_deadline_work(env)

        # Use upper-bound forecast for conservative capacity planning
        upper_work = hi.sum(axis=1) * self.service_mean   # (K,)
        total = current_work + upper_work + near_work
        effective = (1.20 * total[0] + 0.80 * total[1] + 0.60 * total[2])
        desired_cap = effective / max(0.05, self.target_util)
        n_raw = int(np.ceil(desired_cap / self.cap))
        n_raw = int(np.clip(n_raw, env.n_prev - self.ramp,
                            env.n_prev + self.ramp))
        n_raw = int(np.clip(n_raw, self.Nmin, self.Nmax))
        return n_to_action(n_raw, self.cfg)

    # ------------------------------------------------------------------
    def on_step(self, env, info) -> None:
        self.cp.update_after_step(info)

    # ------------------------------------------------------------------
    def metrics(self):
        return {"conformal": self.cp.metrics()}

    # ------------------------------------------------------------------
    def _near_deadline_work(self, env) -> np.ndarray:
        out = np.zeros(env.K)
        for k in range(env.K):
            if not env.queues[k]:
                continue
            near_thresh = max(1, int(np.ceil(0.25 * self.deadlines[k])))
            for task in env.queues[k]:
                slack = task.deadline - env.t + 1
                if slack <= near_thresh:
                    out[k] += task.remaining
        return out
