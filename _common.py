"""
experiments/_common.py
----------------------
Shared utilities for experiment runners. Loads processed data, builds
the env, and exposes day index lists for train / calibration / test.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.data_preprocess import load_processed, day_matrix, normalised_day_matrix
from src.datacenter_env import DataCenterEnv
from src.workload_generator import compute_lambda


def build_env_and_splits(cfg: Dict
                         ) -> Tuple[DataCenterEnv,
                                    Dict[str, List[int]],
                                    np.ndarray]:
    """Construct the environment + train/cal/test day-index lists.

    Returns
    -------
    env : DataCenterEnv (ready to be reset(day_index=...))
    splits : dict[str -> list of day indices into env.day_load_matrix]
    norm_matrix : (D, T) normalised load (used outside the env for
                  forecaster calibration etc.)
    """
    load_df, split_df = load_processed(cfg)
    raw, dates = day_matrix(load_df, cfg["time"]["slots_per_day"])
    norm, _ = normalised_day_matrix(load_df, cfg["time"]["slots_per_day"])
    env = DataCenterEnv(cfg=cfg,
                        day_load_matrix=raw,
                        day_norm_matrix=norm,
                        base_seed=int(cfg.get("seed", 0)))

    # Day-index lookup: which row in env.day_load_matrix corresponds to a date.
    date_to_idx = {d: i for i, d in enumerate(dates)}
    splits = {"train": [], "calibration": [], "test": []}
    for row in split_df.itertuples(index=False):
        d = row.date
        if d in date_to_idx:
            splits[row.split].append(date_to_idx[d])
    return env, splits, norm


def calibration_arrival_data(cfg: Dict,
                             norm_matrix: np.ndarray,
                             cal_days: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (forecast, realised) arrival sequences for calibration days.

    Used to warm-up the conformal learners. We sample fresh Poisson
    realisations using cfg seed offset; deterministic across runs.

    Returns
    -------
    y_hat_cal : (N, K) "predicted" arrival counts (rolling-mean style)
    y_cal     : (N, K) realised Poisson draws
    """
    K = cfg["qos"]["K"]
    T = cfg["time"]["slots_per_day"]
    window = cfg["conformal"]["rolling_window"]
    seed0 = int(cfg.get("seed", 0)) + 1000
    rng_master = np.random.default_rng(seed0)

    all_real, all_hat = [], []
    for d in cal_days:
        lam = compute_lambda(norm_matrix[d], cfg)        # (K, T)
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        real = rng.poisson(lam).astype(np.float64)        # (K, T)
        # Rolling-mean predictions (causal: only past slots).
        hat = np.zeros_like(real)
        for t in range(T):
            if t == 0:
                hat[:, t] = real[:, 0]
            else:
                w = min(window, t)
                hat[:, t] = real[:, max(0, t - w):t].mean(axis=1)
        all_real.append(real.T)   # (T, K)
        all_hat.append(hat.T)
    y_cal = np.concatenate(all_real, axis=0)
    y_hat_cal = np.concatenate(all_hat, axis=0)
    return y_hat_cal, y_cal
