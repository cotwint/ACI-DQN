"""Training/evaluation stats utilities for run_all_experiments.

This module provides lightweight containers used by the monolithic
run_all_experiments.py pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class EpisodeStats:
    """Accumulate per-step metrics for a single episode."""

    def __init__(self, K: int, reward_scale: float) -> None:
        self.K = int(K)
        self.reward_scale = float(reward_scale)
        self.reset()

    def reset(self) -> None:
        self.n_steps = 0
        self.total_energy_cost = 0.0
        self.total_sla_cost = 0.0
        self.total_switching_cost = 0.0
        self.total_cost = 0.0
        self.total_reward = 0.0
        self.total_energy_kwh = 0.0
        self.util_sum = 0.0
        self.n_active_sum = 0.0
        self.completed = np.zeros(self.K, dtype=int)
        self.violations = np.zeros(self.K, dtype=int)
        self.overdue_pending = np.zeros(self.K, dtype=int)

    def step(self,
             t: int,
             active: int,
             util: float,
             it_power_kw: float,
             facility_power_kw: float,
             energy_kwh: float,
             energy_cost: float,
             sla_cost: float,
             switching_cost: float,
             completed: np.ndarray,
             violations: np.ndarray,
             queue_len: np.ndarray,
             backlog_work: np.ndarray,
             overdue_pending: np.ndarray) -> None:
        del t, it_power_kw, facility_power_kw, queue_len, backlog_work

        self.n_steps += 1
        self.total_energy_cost += float(energy_cost)
        self.total_sla_cost += float(sla_cost)
        self.total_switching_cost += float(switching_cost)
        step_cost = float(energy_cost) + float(sla_cost) + float(switching_cost)
        self.total_cost += step_cost
        self.total_reward += -self.reward_scale * step_cost
        self.total_energy_kwh += float(energy_kwh)
        self.util_sum += float(util)
        self.n_active_sum += float(active)
        self.completed += np.asarray(completed, dtype=int)
        self.violations += np.asarray(violations, dtype=int)
        self.overdue_pending += np.asarray(overdue_pending, dtype=int)

    def summarise(self) -> Dict[str, float]:
        denom = max(int(self.completed[0]), 1) if self.K > 0 else 1
        viol_p1 = float(self.violations[0]) / denom if self.K > 0 else 0.0
        n = max(self.n_steps, 1)
        return {
            "total_reward": float(self.total_reward),
            "total_cost": float(self.total_cost),
            "total_energy_cost": float(self.total_energy_cost),
            "total_sla_cost": float(self.total_sla_cost),
            "total_switching_cost": float(self.total_switching_cost),
            "total_energy_kwh": float(self.total_energy_kwh),
            "avg_utilization": float(self.util_sum / n),
            "avg_active_servers": float(self.n_active_sum / n),
            "violation_rate_p1": float(viol_p1),
        }


class TrainingHistory:
    """Lightweight container for per-episode training metrics."""

    def __init__(self) -> None:
        self._rows: List[Dict[str, float]] = []
        self.latest: Optional[Dict[str, float]] = None

    def record(self, episode: int, stats: EpisodeStats,
               extra: Optional[Dict[str, float]] = None) -> None:
        row: Dict[str, float] = {"episode": int(episode)}
        row.update(stats.summarise())
        if extra:
            row.update(extra)
        self._rows.append(row)
        self.latest = row

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._rows)
