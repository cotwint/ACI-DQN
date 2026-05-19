"""
src/rl/train_dqn.py
-------------------
Generic training and evaluation loops shared by ordinary DQN, ACI-DQN
and DtACI-DQN.

The conformal / shield mechanisms are passed in through a
``StateAugmenter`` callable. This avoids three copies of the same loop.

A ``StateAugmenter`` is any object that exposes::

    reset(env, day_index)                 -> None
    augment(state, env) -> np.ndarray     # state used by the agent
    shield(action, state, env) -> int     # post-process the action
    on_step(env, info, action_pre, action_post) -> None
    metrics() -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..datacenter_env import DataCenterEnv
from ..utils import get_logger, ensure_dir
from .dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Pass-through identity augmenter (= ordinary DQN)
# ---------------------------------------------------------------------------

class IdentityAugmenter:
    name = "identity"

    def reset(self, env, day_index):
        pass

    def augment(self, state, env):
        return state

    def shield(self, action, state, env):
        return action

    def on_step(self, env, info, action_pre, action_post):
        pass

    def metrics(self):
        return {}


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    day_index: int
    total_reward: float
    total_cost: float
    elec_cost: float
    qos_cost: float
    switch_cost: float
    sla_violations: np.ndarray
    completed: np.ndarray
    delay_sum: np.ndarray
    avg_n_active: float
    peak_power: float
    avg_power: float
    energy: float
    avg_util: float
    overdue_pending: np.ndarray | None = None  # per-priority overdue pending sum


def rollout_episode(env: DataCenterEnv,
                    agent: DQNAgent,
                    augmenter,
                    day_index: int,
                    train: bool,
                    seed: Optional[int] = None,
                    record_actions: bool = False) -> EpisodeStats:
    # When seed is None, env uses base_seed + day_index (deterministic).
    state, _ = env.reset(day_index, seed=seed)
    augmenter.reset(env, day_index)
    s_aug = augmenter.augment(state, env)

    total_reward = 0.0
    losses: list = []
    action_log: list = []
    while not env.done:
        a_raw = (agent.select_action(s_aug, greedy=not train)
                 if agent is not None else 0)
        a_safe = augmenter.shield(a_raw, s_aug, env)
        next_state, reward, done, info = env.step(a_safe)
        s2_aug = augmenter.augment(next_state, env)
        augmenter.on_step(env, info, a_raw, a_safe)
        if train and agent is not None:
            agent.push(s_aug, a_safe, reward, s2_aug, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        if record_actions:
            action_log.append((a_raw, a_safe, env.history[-1].n_active))
        s_aug = s2_aug
        total_reward += reward

    h = env.history_as_arrays()
    avg_loss = float(np.mean(losses)) if losses else 0.0
    n_grad_updates = len(losses)
    return EpisodeStats(
        day_index=day_index,
        total_reward=total_reward,
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
    ), avg_loss, n_grad_updates, action_log


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_agent(env: DataCenterEnv,
                agent: DQNAgent,
                augmenter,
                train_days: List[int],
                episodes: int,
                rng: np.random.Generator,
                log_path: Optional[str] = None) -> Dict[str, List[float]]:
    log = get_logger("rl.train",
                     log_file=str(log_path) if log_path else None)
    history: Dict[str, List[float]] = {
        "episode": [], "reward": [], "cost": [], "epsilon": [],
        "avg_loss": [], "n_grad_updates": [], "buffer_size": [],
    }
    for ep in range(episodes):
        day = int(rng.choice(train_days))
        stats, avg_loss, n_updates, _ = rollout_episode(
            env, agent, augmenter,
            day_index=day, train=True,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        agent.decay_epsilon()
        history["episode"].append(ep + 1)
        history["reward"].append(stats.total_reward)
        history["cost"].append(stats.total_cost)
        history["epsilon"].append(agent.epsilon)
        history["avg_loss"].append(avg_loss)
        history["n_grad_updates"].append(n_updates)
        history["buffer_size"].append(len(agent.buffer))
        if (ep + 1) % max(1, episodes // 10) == 0 or ep == 0:
            log.info(
                f"  ep {ep+1:4d}/{episodes}  day={day:3d}  "
                f"reward={stats.total_reward:9.1f}  cost={stats.total_cost:9.1f}  "
                f"eps={agent.epsilon:.3f}  loss={avg_loss:.4f}  "
                f"grad_updates={n_updates}  buffer={len(agent.buffer)}"
            )
    return history


def evaluate(env: DataCenterEnv,
             agent: Optional[DQNAgent],
             augmenter,
             eval_days: List[int],
             rng: Optional[np.random.Generator] = None,
             record_actions: bool = False) -> List[EpisodeStats]:
    out: List[EpisodeStats] = []
    all_action_logs: list = []
    for d in eval_days:
        # seed=None -> env uses base_seed + day_index (deterministic),
        # so all methods see identical task traces on each test day.
        stats, _, _, action_log = rollout_episode(
            env, agent, augmenter,
            day_index=int(d),
            train=False,
            seed=None,
            record_actions=record_actions,
        )
        out.append(stats)
        if record_actions:
            all_action_logs.append((d, action_log))
    if record_actions:
        return out, all_action_logs
    return out
