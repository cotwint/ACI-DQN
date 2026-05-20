"""Queue-aware greedy: choose n(t) to serve current backlog, ignoring price."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..datacenter_env import n_to_action


class QueueGreedyPolicy:
    name = "queue_greedy"

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
        work = np.array(
            [sum(task.remaining for task in q) for q in env.queues],
            dtype=np.float64,
        )
        near_work = self._near_deadline_work(env)

        effective = (1.20 * work[0] + 0.80 * work[1] + 0.60 * work[2]
                     + 0.80 * near_work.sum())
        desired_cap = effective / max(0.05, self.target_util)
        n_raw = int(np.ceil(desired_cap / self.cap))
        n_raw = int(np.clip(n_raw, env.n_prev - self.ramp,
                            env.n_prev + self.ramp))
        n_raw = int(np.clip(n_raw, self.Nmin, self.Nmax))
        return n_to_action(n_raw, self.cfg)

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

    def on_step(self, env, info) -> None:
        pass

    def metrics(self):
        return {}

    def warm_up_from_calibration(self, y_hat_cal, y_cal) -> None:
        pass
