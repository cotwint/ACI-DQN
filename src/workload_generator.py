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

# ---------------------------------------------------------------------------
# Workload enhancement functions (deterministic RNG hierarchy)
# Each takes a ``seed: int`` and creates its own ``np.random.default_rng(seed)``.
# No global ``np.random`` is ever used.
# ---------------------------------------------------------------------------

def apply_day_multiplier(lam: np.ndarray, params: Dict, seed: int) -> np.ndarray:
    """Scale all lambdas by a LogNormal daily demand multiplier.

    ``m_day ~ LogNormal(mu=params.mu, sigma=params.sigma)``.

    Returns a *new* array (input is not mutated).
    """
    mu = float(params.get("mu", 0.0))
    sigma = float(params.get("sigma", 0.0))
    if sigma <= 0.0:
        return lam
    rng = np.random.default_rng(seed)
    m = rng.lognormal(mean=mu, sigma=sigma)
    return lam * max(m, 0.1)


def apply_autocorr_noise(lam: np.ndarray, params: Dict, seed: int) -> np.ndarray:
    """Multiplicative AR(1) slot-level noise.

    ``z_t = rho * z_{t-1} + epsilon_t``, ``epsilon_t ~ N(0, sigma^2)``.
    ``lambda_k(t) = lambda_k(t) * exp(z_t)``.

    Returns a *new* array.
    """
    rho = float(params.get("rho", 0.0))
    sigma = float(params.get("sigma", 0.0))
    if rho <= 0.0 and sigma <= 0.0:
        return lam
    T = lam.shape[1]
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=T)
    z = np.zeros(T, dtype=np.float64)
    z[0] = eps[0]
    for t in range(1, T):
        z[t] = rho * z[t - 1] + eps[t]
    return lam * np.exp(z)[None, :]  # broadcast across priorities


def apply_priority_mix_shift(lam: np.ndarray, params: Dict) -> np.ndarray:
    """Shift priority mix during incident hours.

    Modes:
      - ``redistribute``: increase P1 share, reduce P2/P3 proportionally.
        Total load is preserved.
      - ``amplify``: multiply P1 lambda by ``p1_share_factor``, leave others.
        Total load increases.

    Returns a *new* array.
    """
    if not params.get("enabled", False):
        return lam
    mode = params.get("mode", "redistribute")
    factor = float(params.get("p1_share_factor", 1.0))
    hours = params.get("incident_hours", [10, 16])
    slots_per_day = lam.shape[1]
    spph = slots_per_day // 24

    out = lam.copy()
    h0, h1 = int(hours[0]), int(hours[1])
    for h in range(h0, h1):
        t0, t1 = h * spph, (h + 1) * spph
        for t in range(t0, t1):
            if mode == "redistribute":
                # Shift workload from P2,P3 to P1, preserving total
                total = lam[:, t].sum()
                p1_old = lam[0, t]
                p1_new = p1_old * factor
                delta = p1_new - p1_old
                p2p3_total = lam[1, t] + lam[2, t]
                if p2p3_total > 1e-9:
                    out[0, t] = p1_new
                    out[1, t] = max(0.0, lam[1, t] - delta * lam[1, t] / p2p3_total)
                    out[2, t] = max(0.0, lam[2, t] - delta * lam[2, t] / p2p3_total)
                else:
                    out[0, t] = total  # all goes to P1 if P2,P3 empty
            elif mode == "amplify":
                out[0, t] = lam[0, t] * factor
    return np.clip(out, 0.0, None)


def apply_clustered_burst(lam: np.ndarray, params: Dict, seed: int) -> np.ndarray:
    """Markov-chain clustered burst on specified priorities.

    State machine per slot:
      - normal -> burst with probability ``p_start``
      - burst  -> burst with probability ``p_continue``
      - burst  -> normal with probability ``1 - p_continue``

    During burst: ``lambda_k(t) *= multiplier ~ Uniform(min, max)``.

    A minimum burst duration of 2 slots is enforced.

    Returns a *new* array.
    """
    if not params.get("enabled", False):
        return lam
    priorities = params.get("priorities", [3])  # 1-based
    p_start = float(params.get("p_start", 0.03))
    p_continue = float(params.get("p_continue", 0.85))
    mult_min = float(params.get("multiplier_min", 3.0))
    mult_max = float(params.get("multiplier_max", 10.0))

    T = lam.shape[1]
    rng = np.random.default_rng(seed)
    # RNG sub-streams: 0 = state transition, 1 = multiplier draw
    rng_state, rng_mult = rng.spawn(2)

    out = lam.copy()
    in_burst = False
    burst_mult = 1.0
    burst_counter = 0

    for t in range(T):
        if in_burst:
            if burst_counter > 0 or rng_state.random() < p_continue:
                in_burst = True
                if burst_counter == 0:
                    burst_mult = rng_mult.uniform(mult_min, mult_max)
                    burst_counter = 0  # already applied
            else:
                in_burst = False
                burst_mult = 1.0
        else:
            if rng_state.random() < p_start:
                in_burst = True
                burst_mult = rng_mult.uniform(mult_min, mult_max)
                burst_counter = 1  # enforce at least 2-slot burst
            else:
                burst_mult = 1.0

        if burst_counter > 0:
            burst_counter -= 1

        if in_burst and burst_mult > 1.0:
            for pk_1based in priorities:
                k = int(pk_1based) - 1
                if 0 <= k < lam.shape[0]:
                    out[k, t] = lam[k, t] * burst_mult

    return out


# ---------------------------------------------------------------------------
# Enhanced day generation with full RNG hierarchy
# ---------------------------------------------------------------------------

def generate_day_enhanced(x: np.ndarray,
                          cfg: Dict,
                          seed: int,
                          enhancement: Dict | None = None) -> DayWorkload:
    """Sample one day of tasks with optional workload enhancements.

    All randomness is derived from a unified ``seed`` via a deterministic
    RNG hierarchy.  Each enhancement step receives a derived sub-seed so
    that results are fully reproducible.

    Parameters
    ----------
    x : (T,) normalised load curve in [0,1]
    cfg : full config dict
    seed : int master seed
    enhancement : optional workload_enhancement dict (from config/scenario)
    """
    rng = np.random.default_rng(seed)
    sub_seeds = rng.integers(0, 2 ** 31 - 1, size=7)

    slots = cfg["time"]["slots_per_day"]
    K = cfg["qos"]["K"]
    mu = np.asarray(cfg["qos"]["service_mean"], dtype=np.float64)
    sd = np.asarray(cfg["qos"]["service_std"], dtype=np.float64)
    deadlines = np.asarray(cfg["qos"]["deadline_slots"], dtype=int)

    enhancement = enhancement or {}

    # --- Step 1: base lambda ------------------------------------------------
    lam = compute_lambda(x, cfg)

    # --- Step 2: day-level demand multiplier (sub-seed 0) -------------------
    dm = enhancement.get("day_multiplier", {})
    lam = apply_day_multiplier(lam, dm, int(sub_seeds[0]))

    # --- Step 3: autocorrelated slot noise (sub-seed 1) ---------------------
    an = enhancement.get("autocorr_noise", {})
    lam = apply_autocorr_noise(lam, an, int(sub_seeds[1]))

    # --- Step 4: priority mix shift (deterministic given config) ------------
    ps = enhancement.get("priority_mix_shift", {})
    lam = apply_priority_mix_shift(lam, ps)

    # --- Step 5: clustered burst (sub-seeds 2,3) ----------------------------
    cb = enhancement.get("clustered_burst", {})
    lam = apply_clustered_burst(lam, cb, int(sub_seeds[2]))

    # --- Step 6: Poisson arrivals (sub-seed 4) ------------------------------
    rng_arr = np.random.default_rng(int(sub_seeds[4]))
    n_arr = rng_arr.poisson(lam).astype(int)

    # --- Step 7: Work sampling (sub-seed 5) ---------------------------------
    rng_work = np.random.default_rng(int(sub_seeds[5]))

    tasks_per_slot: List[List[List[Task]]] = [
        [[] for _ in range(slots)] for _ in range(K)
    ]
    arrival_work = np.zeros((K, slots), dtype=np.float64)

    for k in range(K):
        for t in range(slots):
            n = int(n_arr[k, t])
            if n == 0:
                continue
            w = rng_work.normal(loc=mu[k], scale=sd[k], size=n)
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


def lambda_matrix_for_days(x_matrix: np.ndarray, cfg: Dict) -> np.ndarray:
    """Compute lambda for every day in the (D,T) normalised load matrix.

    Returns array of shape (D, 3, T).
    """
    D = x_matrix.shape[0]
    out = np.zeros((D, 3, x_matrix.shape[1]), dtype=np.float64)
    for d in range(D):
        out[d] = compute_lambda(x_matrix[d], cfg)
    return out
