"""
src/workload_generator.py
-------------------------
Generate synthetic P1/P2/P3 task workloads driven by the normalised
regional load shape ``x(t)``:

    lambda_1(t) = a1 + b1 * x(t) + c1 * evening(t)
    lambda_2(t) = a2 + b2 * x(t) + c2 * business(t)
    lambda_3(t) = a3 + b3 * night(t) + c3 * (1 - x(t))

``A_k(t) ~ Poisson(lambda_k(t))`` then determines the number of tasks
arriving in each 15-min slot. For each task we sample a compute
requirement c_kj from a clamped normal and assign a deadline equal to
arrival + deadline_slots[k].

NOTE: this is *not* a real Alibaba cluster trace. We use the shape of
regional power load as a proxy for diurnal IT demand only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Indicator builders
# ---------------------------------------------------------------------------

def hour_indicator(slots_per_day: int, hour_range: List[int]) -> np.ndarray:
    """Return a {0,1} indicator over slots for the half-open hour range."""
    h0, h1 = int(hour_range[0]), int(hour_range[1])
    ind = np.zeros(slots_per_day, dtype=np.float64)
    spph = slots_per_day // 24  # slots per hour
    for h in range(h0, h1):
        ind[h * spph:(h + 1) * spph] = 1.0
    return ind


# ---------------------------------------------------------------------------
# Lambda computation
# ---------------------------------------------------------------------------

def compute_lambda(x: np.ndarray, cfg: Dict) -> np.ndarray:
    """Return ``(3, slots_per_day)`` arrival-rate matrix.

    ``x`` is a normalised load curve in [0,1].
    """
    slots = cfg["time"]["slots_per_day"]
    if x.shape[0] != slots:
        raise ValueError(f"x has shape {x.shape}, expected {slots}")

    bh = hour_indicator(slots, cfg["workload"]["business_hours"])
    ev = hour_indicator(slots, cfg["workload"]["evening_hours"])
    ng = hour_indicator(slots, cfg["workload"]["night_hours"])

    p1 = cfg["workload"]["p1"]
    p2 = cfg["workload"]["p2"]
    p3 = cfg["workload"]["p3"]

    lam = np.zeros((3, slots), dtype=np.float64)
    lam[0] = p1["a"] + p1["b"] * x + p1["c"] * ev
    lam[1] = p2["a"] + p2["b"] * x + p2["c"] * bh
    lam[2] = p3["a"] + p3["b"] * ng + p3["c"] * (1.0 - x)
    lam = np.clip(lam, 0.0, None)
    return lam


# ---------------------------------------------------------------------------
# Task list (per priority)
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Single compute task -- mirrors the Octave Q{k} struct."""
    arrival: int    # slot index when task arrived
    deadline: int   # slot index of latest acceptable completion
    work: float     # compute units required
    remaining: float = field(default=0.0)


@dataclass
class DayWorkload:
    """Container for one day's generated workload."""
    lam: np.ndarray                # (3, T) arrival rates
    n_arrivals: np.ndarray         # (3, T) realised task counts
    arrival_work: np.ndarray       # (3, T) total work that arrived that slot
    tasks_per_slot: List[List[List[Task]]]
    # tasks_per_slot[k][t] -> list[Task] arriving at slot t with priority k+1


def generate_day(x: np.ndarray, cfg: Dict, seed: int) -> DayWorkload:
    """Sample one day of tasks. Deterministic given ``seed``.

    Parameters
    ----------
    x : (T,) normalised load curve in [0,1]
    cfg : full config dict
    seed : int seed for numpy Generator
    """
    rng = np.random.default_rng(seed)
    slots = cfg["time"]["slots_per_day"]
    K = cfg["qos"]["K"]
    mu = np.asarray(cfg["qos"]["service_mean"], dtype=np.float64)
    sd = np.asarray(cfg["qos"]["service_std"], dtype=np.float64)
    deadlines = np.asarray(cfg["qos"]["deadline_slots"], dtype=int)

    lam = compute_lambda(x, cfg)            # (3, T)
    n_arr = rng.poisson(lam).astype(int)    # (3, T)

    tasks_per_slot: List[List[List[Task]]] = [
        [[] for _ in range(slots)] for _ in range(K)
    ]
    arrival_work = np.zeros((K, slots), dtype=np.float64)

    for k in range(K):
        for t in range(slots):
            n = int(n_arr[k, t])
            if n == 0:
                continue
            # Clamp work so we never get negatives.
            w = rng.normal(loc=mu[k], scale=sd[k], size=n)
            w = np.clip(w, 0.05, None)
            dl = min(slots - 1, t + int(deadlines[k]))
            for j in range(n):
                tasks_per_slot[k][t].append(
                    Task(arrival=t, deadline=dl,
                         work=float(w[j]), remaining=float(w[j]))
                )
            arrival_work[k, t] = float(w.sum())

    return DayWorkload(
        lam=lam,
        n_arrivals=n_arr,
        arrival_work=arrival_work,
        tasks_per_slot=tasks_per_slot,
    )


# ---------------------------------------------------------------------------
# Helper used by both env and forecaster
# ---------------------------------------------------------------------------

def lambda_matrix_for_days(x_matrix: np.ndarray, cfg: Dict) -> np.ndarray:
    """Compute lambda for every day in the (D,T) normalised load matrix.

    Returns array of shape (D, 3, T).
    """
    D = x_matrix.shape[0]
    out = np.zeros((D, 3, x_matrix.shape[1]), dtype=np.float64)
    for d in range(D):
        out[d] = compute_lambda(x_matrix[d], cfg)
    return out
