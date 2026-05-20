"""
tests/_fixtures.py
------------------
Tiny, fast fixtures used by the unit tests so they don't need the real
CSV nor the heavy training loops.
"""

from __future__ import annotations

import numpy as np


def tiny_cfg() -> dict:
    """A self-contained config dict (no YAML file required)."""
    return {
        "paths": {
            "raw_csv": "",
            "processed_dir": "outputs/processed",
            "figures_dir": "outputs/figures",
            "logs_dir": "outputs/logs",
            "outputs_dir": "outputs",
            "models_dir": "outputs/models",
        },
        "time": {"slots_per_day": 96, "dt_hour": 0.25},
        "server": {"Nmin": 8, "Nmax": 120, "cap_per_server": 1.0,
                   "target_util": 0.75, "ramp_limit": 25},
        "power": {"P_idle": 0.18, "P_peak": 0.42, "P_fixed": 8.0,
                  "PUE": 1.35, "power_gamma": 1.15, "switch_cost": 0.10},
        "qos": {"K": 3,
                "deadline_slots": [2, 8, 32],
                "service_mean":  [0.60, 1.20, 4.00],
                "service_std":   [0.10, 0.25, 0.80],
                "sla_penalty":   [20.0, 8.0, 2.0],
                "overdue_penalty": [0.50, 0.20, 0.05]},
        "price": {"low": 0.32, "middle": 0.55, "high": 0.95,
                  "high_hours":   [10, 11, 14, 15, 19, 20],
                  "middle_hours": [8, 9, 12, 13, 16, 17, 18, 21]},
        "workload": {
            "source": "synthetic",
            "p1": {"a": 1.0, "b": 6.0, "c": 2.0},
            "p2": {"a": 2.0, "b": 4.0, "c": 2.0},
            "p3": {"a": 0.5, "b": 3.5, "c": 2.0},
            "business_hours": [9, 18], "evening_hours": [18, 23],
            "night_hours": [0, 6], "seed_offset": 0,
        },
        "workload_enhancement": {
            "day_multiplier": {"mu": 0.0, "sigma": 0.0},
            "autocorr_noise": {"rho": 0.0, "sigma": 0.0},
            "clustered_burst": {
                "enabled": False, "priorities": [3],
                "p_start": 0.03, "p_continue": 0.85,
                "multiplier_min": 3.0, "multiplier_max": 10.0,
            },
            "priority_mix_shift": {
                "enabled": False, "mode": "redistribute",
                "p1_share_factor": 1.0, "incident_hours": [10, 16],
            },
        },
        "split": {"train_ratio": 0.6, "cal_ratio": 0.2, "test_ratio": 0.2},
        "conformal": {
            "alpha": 0.10, "horizon": 4,
            "alpha_min": 0.005, "alpha_max": 0.30, "aci_eta": 0.05,
            "dtaci_etas": [0.005, 0.02, 0.05, 0.1, 0.25],
            "dtaci_sigma": 0.10, "dtaci_meta_lr": 0.10,
            "protect_priorities": [1, 2],
            "forecaster": "rolling_mean", "rolling_window": 8,
            "upper_bound_clip_method": "p95",
            "lambda_scale": 20.0,
        },
        "rl": {"algorithm": "DQN", "action_bins": 21, "gamma": 0.99,
               "lr": 5e-4, "batch_size": 32, "replay_size": 1024,
               "train_episodes": 2, "eval_episodes": 2,
               "epsilon_start": 1.0, "epsilon_end": 0.05,
               "epsilon_decay": 0.99, "target_update_interval": 50,
               "learning_starts": 50, "hidden_sizes": [32, 32],
               "reward_scale": 0.01, "max_grad_norm": 5.0,
               "reward_weights": {"elec": 1.0, "qos": 5.0, "switch": 0.2},
               "reward_scales": {"elec": 10.0, "qos": 10.0, "switch": 5.0}},
        "experiment": {
            "training_seeds": [2024],
            "scenario_seeds": [100],
        },
        "diagnostics": {"rolling_miscoverage_window": 24},
        "seed": 7,
        "verbose": False,
    }


def tiny_load_matrices(D: int = 3, T: int = 96, seed: int = 0):
    """Synthetic (D,T) load matrices: raw + normalised in [0,1]."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(100, 500, size=(D, T))
    raw[:, 30:60] *= 1.5
    nmin = raw.min(axis=1, keepdims=True)
    nmax = raw.max(axis=1, keepdims=True)
    norm = (raw - nmin) / np.maximum(nmax - nmin, 1e-9)
    return raw, norm
