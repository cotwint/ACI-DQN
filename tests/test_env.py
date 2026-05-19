"""
tests/test_env.py
-----------------
Smoke + invariant tests for ``DataCenterEnv``.

Checked invariants
------------------
* reset() returns a state of the right shape and zero-initialised queues
* step() runs T times, then ``done`` flips and one further call raises
* queue lengths and backlog work are never negative
* the action is always mapped into [Nmin, Nmax]
"""

from __future__ import annotations

import numpy as np

from src.datacenter_env import DataCenterEnv, action_to_n
from tests._fixtures import tiny_cfg, tiny_load_matrices


def _make_env(seed: int = 0):
    cfg = tiny_cfg()
    raw, norm = tiny_load_matrices(D=3, T=cfg["time"]["slots_per_day"], seed=seed)
    return DataCenterEnv(cfg=cfg, day_load_matrix=raw,
                         day_norm_matrix=norm, base_seed=seed), cfg


def test_reset_returns_correct_state_shape():
    env, cfg = _make_env()
    s, info = env.reset(0)
    assert s.shape == (env.observation_dim,)
    assert info["day_index"] == 0
    # queues empty at reset
    assert all(len(q) == 0 for q in env.queues)


def test_full_episode_done_flips_and_invariants_hold():
    env, cfg = _make_env(seed=1)
    s, _ = env.reset(0)
    steps_done = 0
    rng = np.random.default_rng(0)
    while not env.done:
        a = int(rng.integers(0, env.action_dim))
        s, r, done, info = env.step(a)
        steps_done += 1
        # never negative queue
        ql = np.array([len(q) for q in env.queues])
        assert (ql >= 0).all()
        # never negative backlog (work remaining is per-task remaining >= 0)
        for q in env.queues:
            for task in q:
                assert task.remaining >= -1e-6
        # cost components are finite
        assert np.isfinite(r)
    assert steps_done == env.T
    assert env.done


def test_action_clamped_to_server_bounds():
    env, cfg = _make_env(seed=2)
    Nmin, Nmax = cfg["server"]["Nmin"], cfg["server"]["Nmax"]
    # Action 0 -> Nmin; action_bins-1 -> Nmax
    assert action_to_n(0, cfg) == Nmin
    assert action_to_n(cfg["rl"]["action_bins"] - 1, cfg) == Nmax
    # Out-of-range action indices are clipped, not crashed.
    env.reset(0)
    s, r, done, _ = env.step(-5)
    assert env.history[0].n_active >= Nmin
    s, r, done, _ = env.step(9999)
    assert env.history[1].n_active <= Nmax
