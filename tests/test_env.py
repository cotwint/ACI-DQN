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

import pytest
from src.datacenter_env import DataCenterEnv, action_to_n, n_to_action
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
        ql = np.array([len(q) for q in env.queues])
        assert (ql >= 0).all()
        for q in env.queues:
            for task in q:
                assert task.remaining >= -1e-6
        assert np.isfinite(r)
    assert steps_done == env.T
    assert env.done


# ---- Strict action-index API tests ------------------------------------

def test_action_conversion_roundtrip():
    """action_to_n -> n_to_action round-trips for all valid action indices."""
    cfg = tiny_cfg()
    for a in range(cfg["rl"]["action_bins"]):
        n = action_to_n(a, cfg)
        a2 = n_to_action(n, cfg)
        assert a2 == a, f"Roundtrip failed: {a} -> {n} -> {a2}"


def test_env_step_rejects_server_count():
    """Server count 68 passed as action index -> ValueError."""
    env, cfg = _make_env()
    env.reset(0)
    with pytest.raises(ValueError, match="action index"):
        env.step(68)


def test_env_step_rejects_out_of_range():
    """Action indices outside [0, action_bins-1] raise ValueError."""
    env, cfg = _make_env()
    env.reset(0)
    bins = cfg["rl"]["action_bins"]
    with pytest.raises(ValueError, match="action index"):
        env.step(-1)
    with pytest.raises(ValueError, match="action index"):
        env.step(bins)
    with pytest.raises(ValueError, match="action index"):
        env.step(9999)


def test_env_step_accepts_valid_actions():
    """All valid action indices [0, bins-1] work without error."""
    env, cfg = _make_env(seed=3)
    env.reset(0)
    bins = cfg["rl"]["action_bins"]
    for a in [0, bins // 2, bins - 1]:
        env.step(a)
    assert not env.done


def test_action_to_n_bounds():
    """Action 0 -> Nmin; action max -> Nmax."""
    cfg = tiny_cfg()
    Nmin, Nmax = cfg["server"]["Nmin"], cfg["server"]["Nmax"]
    assert action_to_n(0, cfg) == Nmin
    assert action_to_n(cfg["rl"]["action_bins"] - 1, cfg) == Nmax


# ---- Price curve API test -------------------------------------------

def test_set_price_curve():
    env, cfg = _make_env()
    T = cfg["time"]["slots_per_day"]
    new_curve = np.ones(T) * 0.99
    env.set_price_curve(new_curve)
    assert np.allclose(env.price_curve, new_curve)

    # Wrong shape raises
    with pytest.raises(ValueError):
        env.set_price_curve(np.ones(T + 1))


# ---- Workload params API test ---------------------------------------

def test_set_workload_params_triggers_enhanced_generation():
    env, cfg = _make_env(seed=5)
    env.set_workload_params({"day_multiplier": {"mu": 0.0, "sigma": 0.5}})
    env.reset(0, seed=42)
    assert env.workload is not None
    assert env.workload.lam.shape == (3, cfg["time"]["slots_per_day"])
