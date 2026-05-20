"""
experiments/_heuristic_runner.py
--------------------------------
Runs a heuristic policy (Fixed / QueueGreedy / PriceAware / ForecastGreedy /
ConformalGreedy) for a given list of day indices.

Policies follow the unified protocol:
    reset() / act(state, env) / on_step(env, info) / metrics()
    warm_up_from_calibration(y_hat, y_cal)  [optional, no-op for simple policies]
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.datacenter_env import DataCenterEnv
from src.rl.train_dqn import EpisodeStats


def run_heuristic(env: DataCenterEnv,
                  policy,
                  day_indices: List[int],
                  rng: Optional[np.random.Generator] = None) -> List[EpisodeStats]:
    out: List[EpisodeStats] = []
    for d in day_indices:
        # Use deterministic seed (base_seed + day_index) so all methods
        # see identical task traces for fair comparison.
        state, _ = env.reset(int(d))
        policy.reset()
        while not env.done:
            action = policy.act(state, env)
            next_state, _, _, info = env.step(action)
            policy.on_step(env, info)
            state = next_state
        h = env.history_as_arrays()
        out.append(EpisodeStats(
            day_index=int(d),
            total_reward=-float(h["total_cost"].sum()),
            total_cost=float(h["total_cost"].sum()),
            elec_cost=float(h["elec_cost"].sum()),
            qos_cost=float(h["qos_cost"].sum()),
            switch_cost=float(h["switch_cost"].sum()),
            sla_violations=h["violations"].sum(axis=0),
            completed=h["completed"].sum(axis=0),
            delay_sum=h["delay_sum"].sum(axis=0),
            avg_n_active=float(h["n_active"].mean()),
            peak_power=float(h["facility_power_kw"].max()),
            avg_power=float(h["facility_power_kw"].mean()),
            energy=float(h["energy_kwh"].sum()),
            avg_util=float(h["util"].mean()),
            overdue_pending=h.get("qos_cost_per_priority", np.zeros((0, 3))).sum(axis=0),
        ))
    return out
