"""Price-aware greedy: defers P3 when price is high (port of greedy_policy.m)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..datacenter_env import n_to_action


class PriceAwareGreedyPolicy:
    name = "price_aware_greedy"

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.Nmin = int(cfg["server"]["Nmin"])
        self.Nmax = int(cfg["server"]["Nmax"])
        self.cap = float(cfg["server"]["cap_per_server"])
        self.target_util = float(cfg["server"]["target_util"])
        self.ramp = int(cfg["server"]["ramp_limit"])
        self.deadlines = np.asarray(cfg["qos"]["deadline_slots"], dtype=int)

    def reset(self):
        pass

    def act(self, state: np.ndarray, env) -> int:
        K = env.K
        work = np.zeros(K)
        near_work = np.zeros(K)
        min_slack = np.full(K, np.inf)

        for k in range(K):
            if not env.queues[k]:
                continue
            slacks = np.array([task.deadline - env.t + 1
                              for task in env.queues[k]])
            work[k] = sum(task.remaining for task in env.queues[k])
            min_slack[k] = float(slacks.min())
            near_thresh = max(1, int(np.ceil(0.25 * self.deadlines[k])))
            mask = slacks <= near_thresh
            near_work[k] = sum(task.remaining
                               for task, m in zip(env.queues[k], mask) if m)

        price = float(env.price_curve[env.t])
        price_mean = float(env.price_curve.mean())
        price_std = float(env.price_curve.std())

        if price > price_mean + 0.25 * price_std:
            deferrable = 0.25
        else:
            deferrable = 1.00
        if min_slack[2] <= 4:
            deferrable = 1.00

        effective = (1.20 * work[0]
                     + 0.80 * work[1]
                     + deferrable * 0.60 * work[2]
                     + 0.80 * near_work.sum())
        desired_cap = effective / max(0.05, self.target_util)
        n_raw = int(np.ceil(desired_cap / self.cap))
        n_raw = int(np.clip(n_raw, env.n_prev - self.ramp,
                            env.n_prev + self.ramp))
        n_raw = int(np.clip(n_raw, self.Nmin, self.Nmax))
        return n_to_action(n_raw, self.cfg)

    def on_step(self, env, info) -> None:
        pass

    def metrics(self):
        return {}

    def warm_up_from_calibration(self, y_hat_cal, y_cal) -> None:
        pass
