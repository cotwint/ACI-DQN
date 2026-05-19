"""
experiments/_heuristic_runner.py
--------------------------------
Runs a heuristic policy (Fixed / QueueGreedy / PriceAware) for a given
list of day indices. Mirrors ``rollout_episode`` from ``train_dqn.py``
but without the DQNAgent.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.datacenter_env import DataCenterEnv
from src.rl.train_dqn import EpisodeStats


def run_heuristic(env: DataCenterEnv,
                  policy,
                  day_indices: List[int],
                  rng: np.random.Generator) -> List[EpisodeStats]:
    out: List[EpisodeStats] = []
    for d in day_indices:
        # Use deterministic seed (base_seed + day_index) so all methods
        # see identical task traces for fair comparison.
        state, _ = env.reset(int(d))
        policy.reset()
        while not env.done:
            action = policy.act(state, env)
            state, _, _, _ = env.step(action)
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
