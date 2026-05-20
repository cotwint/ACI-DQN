"""Forecast-Greedy: uses rolling-mean point forecast of next H steps for
capacity planning. Same base forecaster as Forecast-DQN / Conformal-DQN.

Does NOT use conformal intervals — only the point forecast.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..conformal.forecaster import build_forecaster
from ..datacenter_env import n_to_action


class ForecastGreedyPolicy:
    name = "forecast_greedy"

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

        # Same forecaster as Forecast-DQN / ACI-DQN
        kind = cfg["conformal"]["forecaster"]
        window = int(cfg["conformal"]["rolling_window"])
        self.forecasters = [
            build_forecaster(kind, window=window, horizon=self.H)
            for _ in range(self.K)
        ]
        self.history: list = []   # per-slot (K,) arrival counts

    # ------------------------------------------------------------------
    def reset(self):
        self.history = []

    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, env) -> int:
        # Rolling-mean forecast of next H steps for each priority
        forecast = self._forecast_next_H()   # (K, H)
        # Current work
        current_work = np.array(
            [sum(task.remaining for task in q) for q in env.queues],
            dtype=np.float64,
        )
        near_work = self._near_deadline_work(env)

        # Expected future work over H slots
        future_work = (forecast.sum(axis=1) * self.service_mean)   # (K,)
        total = current_work + future_work + near_work
        effective = (1.20 * total[0] + 0.80 * total[1] + 0.60 * total[2])
        desired_cap = effective / max(0.05, self.target_util)
        n_raw = int(np.ceil(desired_cap / self.cap))
        n_raw = int(np.clip(n_raw, env.n_prev - self.ramp,
                            env.n_prev + self.ramp))
        n_raw = int(np.clip(n_raw, self.Nmin, self.Nmax))
        return n_to_action(n_raw, self.cfg)

    # ------------------------------------------------------------------
    def on_step(self, env, info) -> None:
        arrivals = np.asarray(info["arrivals"], dtype=np.float64)
        self.history.append(arrivals)

    # ------------------------------------------------------------------
    def metrics(self):
        return {}

    # ------------------------------------------------------------------
    def warm_up_from_calibration(self, y_hat_cal, y_cal) -> None:
        pass

    # ------------------------------------------------------------------
    def _forecast_next_H(self) -> np.ndarray:
        """Return (K, H) point-forecast matrix for next H steps."""
        out = np.zeros((self.K, self.H), dtype=np.float64)
        for k in range(self.K):
            hist_k = np.array([h[k] for h in self.history], dtype=np.float64)
            for h in range(1, self.H + 1):
                y_hat = self.forecasters[k].predict(hist_k, h=h)
                # predict() may return a scalar or 1-element array
                out[k, h - 1] = float(y_hat) if np.ndim(y_hat) == 0 else float(y_hat[0])
        return out

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
