"""
stress_test_scenarios.py
------------------------
Three high-pressure scenarios to evaluate RL + conformal methods under
distribution shift, burst arrivals, and combined extreme stress.

Scenarios:
  S1 - "Seasonal Peak Shift": Systematically shifted arrival rates.
       Forecaster calibrated on normal data will under-predict.
  S2 - "Burst Anomaly": Unpredictable P3 bursts with short deadlines.
       Tests uncertainty quantification when forecasts are unreliable.
  S3 - "Extreme Coincident": All stressors + 5x price spikes.
       The "perfect storm" that should cause SLA violations.

Usage:
    python stress_test_scenarios.py
"""

import sys; sys.path.insert(0, '.')
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils import load_config, set_global_seed, ensure_dir
from src.data_preprocess import day_matrix, normalised_day_matrix
from src.datacenter_env import DataCenterEnv, action_to_n, n_to_action
from src.workload_generator import Task, DayWorkload
from src.rl.dqn_agent import DQNAgent
from src.rl.train_dqn import IdentityAugmenter, EpisodeStats
from src.rl.augmenters import ConformalAugmenter
from src.baselines.fixed_policy import FixedPolicy
from src.baselines.queue_greedy_policy import QueueGreedyPolicy
from src.baselines.price_aware_greedy_policy import PriceAwareGreedyPolicy

cfg = load_config("config.yaml")
set_global_seed(2024)
OUT_DIR = ensure_dir("outputs/stress_test")
T = cfg["time"]["slots_per_day"]
K = cfg["qos"]["K"]
Nmin = cfg["server"]["Nmin"]
Nmax = cfg["server"]["Nmax"]
deadlines = np.array(cfg["qos"]["deadline_slots"])
service_mean = np.array(cfg["qos"]["service_mean"])
service_std = np.array(cfg["qos"]["service_std"])
cap = cfg["server"]["cap_per_server"]

# Load the processed load data for x(t) curves
processed = pd.read_csv("outputs/processed/processed_load.csv")
day_mat_raw, _ = day_matrix(processed, T)
day_mat_norm, _ = normalised_day_matrix(processed, T)


def make_env():
    """Fresh env with the real x(t) curves."""
    return DataCenterEnv(cfg=cfg, day_load_matrix=day_mat_raw,
                          day_norm_matrix=day_mat_norm, base_seed=2024)


def build_workload(arrival_rates_fn, burst_cfg=None, rng_seed=9999):
    """
    Build a DayWorkload from custom arrival rate functions.

    arrival_rates_fn(t, k) -> float: expected arrivals for slot t, priority k.
    burst_cfg: dict or None, with keys 'prob', 'multiplier_range', 'deadline_frac'.
    """
    rng = np.random.default_rng(rng_seed)
    tasks_per_slot = [[[] for _ in range(T)] for _ in range(K)]
    arrival_work = np.zeros((K, T), dtype=np.float64)
    burst = burst_cfg or {}

    total_tasks = 0
    total_p3 = 0
    burst_count = 0

    for t in range(T):
        for k in range(K):
            lam = arrival_rates_fn(t, k)

            # Apply random burst injection for P3
            if k == 2 and burst and rng.random() < burst.get('prob', 0.0):
                lam *= rng.uniform(*burst.get('multiplier_range', (3, 8)))
                burst_count += 1

            n_arrive = rng.poisson(max(0.0, lam))
            if n_arrive == 0:
                continue

            work_vals = rng.normal(loc=service_mean[k], scale=service_std[k],
                                    size=n_arrive)
            work_vals = np.clip(work_vals, 0.05, None)

            # High randomness work: use log-normal for P3 during stress
            if k == 2 and burst:
                work_vals = rng.lognormal(
                    mean=np.log(service_mean[k]),
                    sigma=burst.get('work_sigma', 0.8),
                    size=n_arrive)
                work_vals = np.clip(work_vals, 0.05, service_mean[k] * 8)

            for j in range(n_arrive):
                # Shorter deadlines during bursts
                if k == 2 and burst and rng.random() < burst.get('tight_deadline_prob', 0.6):
                    d_slot = t + int(rng.integers(
                        1, max(2, int(deadlines[k] * burst.get('deadline_frac', 0.15))) + 1))
                else:
                    d_slot = t + int(deadlines[k])
                d_slot = min(T - 1, max(t + 1, d_slot))

                tasks_per_slot[k][t].append(
                    Task(arrival=t, deadline=d_slot,
                         work=float(work_vals[j]), remaining=float(work_vals[j])))
                total_tasks += 1
                if k == 2:
                    total_p3 += 1
            arrival_work[k, t] = float(work_vals.sum())

    return DayWorkload(
        lam=np.zeros((K, T)),
        n_arrivals=np.array([[len(tasks_per_slot[k][t]) for t in range(T)]
                             for k in range(K)]),
        arrival_work=arrival_work,
        tasks_per_slot=tasks_per_slot,
    ), total_tasks, total_p3, burst_count


def fresh_workload(workload_template):
    """Deep-copy a workload so each policy gets fresh Task objects."""
    tsp = [[[] for _ in range(T)] for _ in range(K)]
    for k in range(K):
        for t in range(T):
            for task in workload_template.tasks_per_slot[k][t]:
                tsp[k][t].append(Task(
                    arrival=task.arrival, deadline=task.deadline,
                    work=task.work, remaining=task.work))
    return DayWorkload(
        lam=workload_template.lam.copy(),
        n_arrivals=workload_template.n_arrivals.copy(),
        arrival_work=workload_template.arrival_work.copy(),
        tasks_per_slot=tsp,
    )


def make_price_curve(scenario_price_multipliers=None):
    """Build a custom electricity price curve."""
    base = np.full(T, 0.5)  # base price CNY/kWh
    # Diurnal pattern
    for t in range(T):
        hour = (t * 15) // 60
        if 8 <= hour < 12:
            base[t] = 0.7
        elif 12 <= hour < 17:
            base[t] = 0.6
        elif 17 <= hour < 21:
            base[t] = 0.9
        elif 21 <= hour < 23:
            base[t] = 0.7
    if scenario_price_multipliers:
        for t_range, mult in scenario_price_multipliers:
            base[t_range[0]:t_range[1]] *= mult
    return base


def run_heuristic(env, policy, workload, price_curve):
    """Run one heuristic episode and return stats dict."""
    env.reset(0)
    env.workload = workload
    env.set_price_curve(price_curve)
    policy.reset()
    while not env.done:
        action = policy.act(env.get_state(), env)
        a_idx = n_to_action(action, cfg)
        env.step(a_idx)
    return _collect_stats(env)


def run_rl(env, agent, aug, workload, price_curve):
    """Run one RL episode and return stats dict."""
    state, _ = env.reset(0)
    env.workload = workload
    env.set_price_curve(price_curve)
    aug.reset(env, 0)
    while not env.done:
        s_aug = aug.augment(state, env)
        a_raw = agent.select_action(s_aug, greedy=True)
        a_n = action_to_n(a_raw, cfg)
        a_safe = aug.shield(a_n, s_aug, env)
        a_safe_idx = n_to_action(a_safe, cfg)
        next_state, _, _, info = env.step(a_safe_idx)
        aug.on_step(env, info, a_raw, a_safe)
        state = next_state
    return _collect_stats(env)


def _collect_stats(env):
    h = env.history_as_arrays()
    return {
        'total_cost': float(h['total_cost'].sum()),
        'elec_cost': float(h['elec_cost'].sum()),
        'qos_cost': float(h['qos_cost'].sum()),
        'switch_cost': float(h['switch_cost'].sum()),
        'avg_servers': float(h['n_active'].mean()),
        'avg_util': float(h['util'].mean()),
        'peak_power': float(h['facility_power_kw'].max()),
        'energy': float(h['energy_kwh'].sum()),
        'P1_violations': int(h['violations'].sum(axis=0)[0]),
        'P2_violations': int(h['violations'].sum(axis=0)[1]),
        'P3_violations': int(h['violations'].sum(axis=0)[2]),
        'P1_completed': int(h['completed'].sum(axis=0)[0]),
        'P2_completed': int(h['completed'].sum(axis=0)[1]),
        'P3_completed': int(h['completed'].sum(axis=0)[2]),
        'total_P1': int(h['arrivals'].sum(axis=0)[0]),
        'total_P2': int(h['arrivals'].sum(axis=0)[1]),
        'total_P3': int(h['arrivals'].sum(axis=0)[2]),
    }


# =========================================================================
# Load pre-trained RL agents
# =========================================================================
def load_agents():
    agents = {}
    augmenters = {}

    # Plain DQN
    dqn = DQNAgent(
        state_dim=17, action_dim=cfg['rl']['action_bins'],
        hidden=cfg['rl']['hidden_sizes'], lr=cfg['rl']['lr'],
        batch_size=cfg['rl']['batch_size'], replay_size=cfg['rl']['replay_size'],
        epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.995,
        target_update_interval=cfg['rl']['target_update_interval'],
        learning_starts=cfg['rl']['learning_starts'],
        reward_scale=cfg['rl']['reward_scale'],
        max_grad_norm=cfg['rl']['max_grad_norm'],
    )
    dqn.load('outputs/models/dqn.pt')
    dqn.epsilon = 0.0
    agents['dqn'] = dqn
    augmenters['dqn'] = IdentityAugmenter()

    # ACI-DQN
    aci = DQNAgent(
        state_dim=17 + 2 * K, action_dim=cfg['rl']['action_bins'],
        hidden=cfg['rl']['hidden_sizes'], lr=cfg['rl']['lr'],
        batch_size=cfg['rl']['batch_size'], replay_size=cfg['rl']['replay_size'],
        epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.995,
        target_update_interval=cfg['rl']['target_update_interval'],
        learning_starts=cfg['rl']['learning_starts'],
        reward_scale=cfg['rl']['reward_scale'],
        max_grad_norm=cfg['rl']['max_grad_norm'],
    )
    aci.load('outputs/models/aci_dqn.pt')
    aci.epsilon = 0.0
    agents['aci_dqn'] = aci
    augmenters['aci_dqn'] = ConformalAugmenter(cfg, learner='aci', use_shield=False)

    return agents, augmenters


# =========================================================================
# SCENARIO 1: Seasonal Peak Shift (distribution shift)
# =========================================================================
def scenario_1():
    """
    All arrival rates systematically increased + diurnal pattern shifted.
    Forecaster was calibrated on normal patterns -> will under-predict.
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Seasonal Peak Shift")
    print("="*70)

    def arrival_rates(t, k):
        # Shifted parameters compared to normal:
        # Normal: P1: 1.0+6.0*x+2.0*ev, P2: 2.0+4.0*x+2.0*bh, P3: 0.5+3.5*ng+2.0*(1-x)
        hour = (t * 15) // 60
        is_evening = 1.0 if 18 <= hour < 23 else 0.0
        is_business = 1.0 if 9 <= hour < 18 else 0.0
        is_night = 1.0 if 0 <= hour < 6 else 0.0
        x = 0.6  # typical load level

        if k == 0:
            return 2.5 + 8.0 * x + 4.0 * is_evening   # was 1.0 + 6.0*x + 2.0*ev
        elif k == 1:
            return 4.0 + 6.0 * x + 4.0 * is_business    # was 2.0 + 4.0*x + 2.0*bh
        else:
            return 3.0 + 5.0 * is_night + 4.0 * (1.0 - x)  # was 0.5 + 3.5*ng + 2.0*(1-x)

    wl, total, p3_total, _ = build_workload(arrival_rates, rng_seed=1000)
    price = make_price_curve()

    arrivals_p3 = sum(1 for k in range(K) for t in range(T)
                      for _ in wl.tasks_per_slot[k][t] if k == 2)

    print(f"  Total tasks: {total}, P3 tasks: {arrivals_p3}")
    print(f"  Expected optimal servers: ~{arrivals_p3 * service_mean[2] / (cap * 0.85 * T):.0f}")

    return wl, price, f"S1_SeasonalShift_{total}tasks"


# =========================================================================
# SCENARIO 2: Burst Anomaly (high randomness)
# =========================================================================
def scenario_2():
    """
    Base rates normal, but 20% of slots get 5-15x P3 bursts with
    tight deadlines (1-4 slots). High work variance via log-normal.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Burst Anomaly")
    print("="*70)

    def arrival_rates(t, k):
        hour = (t * 15) // 60
        if k == 0:
            return 3.0  # moderate P1
        elif k == 1:
            return 4.0  # moderate P2
        else:
            return 2.0  # baseline P3 (bursts injected separately)

    burst_cfg = {
        'prob': 0.20,                 # 20% of P3 slots get burst
        'multiplier_range': (5, 15),  # 5-15x normal P3 arrivals
        'deadline_frac': 0.10,        # deadlines at 10% of normal (3 slots)
        'tight_deadline_prob': 0.80,  # 80% of burst tasks have tight deadline
        'work_sigma': 1.2,            # high work variance
    }

    wl, total, p3_total, burst_slots = build_workload(
        arrival_rates, burst_cfg=burst_cfg, rng_seed=2000)
    price = make_price_curve([
        (np.arange(68, 81), 2.0),  # moderate price bump during evening
    ])

    print(f"  Total tasks: {total}, P3 tasks: {p3_total}")
    print(f"  P3 burst slots: {burst_slots}")
    print(f"  P3 tasks with tight deadline: ~{p3_total * 0.8 * 0.2:.0f}")

    return wl, price, f"S2_BurstAnomaly_{total}tasks"


# =========================================================================
# SCENARIO 3: Extreme Coincident Stress
# =========================================================================
def scenario_3():
    """
    All factors combined:
    - 2x normal arrival rates across all priorities
    - 25% of slots have P3 bursts (8-20x)
    - Very short P3 deadlines (10-20% of normal)
    - 5x electricity price during evening peak (slots 68-80)
    - P1/P2 also elevated during burst windows
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Extreme Coincident Stress")
    print("="*70)

    def arrival_rates(t, k):
        hour = (t * 15) // 60
        is_evening = 1.0 if 18 <= hour < 23 else 0.0
        is_business = 1.0 if 9 <= hour < 18 else 0.0
        x = 0.65

        if k == 0:
            return 3.0 + 10.0 * x + 5.0 * is_evening
        elif k == 1:
            return 5.0 + 8.0 * x + 5.0 * is_business
        else:
            return 4.0 + 6.0 * (1.0 - x)  # P3 is heavy overall

    burst_cfg = {
        'prob': 0.25,
        'multiplier_range': (8, 20),
        'deadline_frac': 0.08,        # 2-3 slot deadlines during burst
        'tight_deadline_prob': 0.90,
        'work_sigma': 1.5,
    }

    wl, total, p3_total, burst_slots = build_workload(
        arrival_rates, burst_cfg=burst_cfg, rng_seed=3000)
    price = make_price_curve([
        (np.arange(68, 81), 5.0),   # 5x price spike evening peak
        (np.arange(40, 49), 2.0),   # 2x afternoon bump
    ])

    print(f"  Total tasks: {total}, P3 tasks: {p3_total}")
    print(f"  P3 burst slots: {burst_slots}")
    print(f"  5x price window: slots 68-80")

    return wl, price, f"S3_ExtremeCoincident_{total}tasks"


# =========================================================================
# Run all methods on a scenario
# =========================================================================
def evaluate_scenario(scenario_fn, agents, augmenters):
    wl_template, price_curve, label = scenario_fn()

    policies = {
        'fixed': FixedPolicy(cfg),
        'queue_greedy': QueueGreedyPolicy(cfg),
        'price_aware_greedy': PriceAwareGreedyPolicy(cfg),
    }

    results = []
    env = make_env()

    # --- Heuristics ---
    for mkey, policy in policies.items():
        wl = fresh_workload(wl_template)
        env_copy = make_env()
        stats = run_heuristic(env_copy, policy, wl, price_curve)
        stats['method'] = mkey
        stats['scenario'] = label
        results.append(stats)
        print(f"  {mkey:25s}: cost={stats['total_cost']:8.1f}, "
              f"servers={stats['avg_servers']:5.1f}, util={stats['avg_util']:.3f}, "
              f"P3_viol={stats['P3_violations']}, P3_comp={stats['P3_completed']}/{stats['total_P3']}")

    # --- RL methods ---
    for mkey in ['dqn', 'aci_dqn']:
        wl = fresh_workload(wl_template)
        env_copy = make_env()
        agent = agents[mkey]
        aug = augmenters[mkey]
        # For conformal methods, re-warm calibrator per fresh augmenter
        if mkey != 'dqn':
            aug = ConformalAugmenter(
                cfg, learner='aci', use_shield=False)
        stats = run_rl(env_copy, agent, aug, wl, price_curve)
        stats['method'] = mkey
        stats['scenario'] = label
        results.append(stats)
        print(f"  {mkey:25s}: cost={stats['total_cost']:8.1f}, "
              f"servers={stats['avg_servers']:5.1f}, util={stats['avg_util']:.3f}, "
              f"P3_viol={stats['P3_violations']}, P3_comp={stats['P3_completed']}/{stats['total_P3']}")

    return pd.DataFrame(results)


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("Loading pre-trained models...")
    agents, augmenters = load_agents()
    print("  Models loaded.\n")

    all_results = []

    for scenario_fn in [scenario_1, scenario_2, scenario_3]:
        df = evaluate_scenario(scenario_fn, agents, augmenters)
        all_results.append(df)

    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv(OUT_DIR / "stress_scenario_results.csv", index=False)

    # ---- Summary tables ----
    print("\n" + "="*70)
    print("SUMMARY: Cost Comparison (CNY)")
    print("="*70)
    methods = ['fixed', 'queue_greedy', 'price_aware_greedy',
               'dqn', 'aci_dqn']
    labels  = ['Fixed', 'Queue-Greedy', 'Price-Greedy',
               'DQN', 'ACI-DQN']

    for scenario in full_df['scenario'].unique():
        sdf = full_df[full_df['scenario'] == scenario]
        print(f"\n  {scenario}:")
        print(f"  {'Method':20s} {'Cost':>10s} {'Servers':>8s} {'Util':>8s} "
              f"{'P3_Viol':>8s} {'P3_Comp':>8s}")
        print(f"  {'-'*60}")
        for m, lbl in zip(methods, labels):
            row = sdf[sdf['method'] == m]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            print(f"  {lbl:20s} {r['total_cost']:10.1f} {r['avg_servers']:8.1f} "
                  f"{r['avg_util']:8.3f} {int(r['P3_violations']):8d} "
                  f"{int(r['P3_completed']):5d}/{int(r['total_P3']):d}")

    # ---- Key metric: P3 SLA violation differentiation ----
    print("\n" + "="*70)
    print("ANALYSIS: P3 SLA Violation Rates Across Scenarios")
    print("="*70)
    for scenario in full_df['scenario'].unique():
        sdf = full_df[full_df['scenario'] == scenario]
        print(f"\n  {scenario}:")
        for m, lbl in zip(methods, labels):
            row = sdf[sdf['method'] == m]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            viol_rate = r['P3_violations'] / max(1, r['total_P3']) * 100
            bar = '█' * int(viol_rate / 2) if viol_rate > 0 else '·'
            print(f"  {lbl:20s} {viol_rate:6.2f}% {bar}")

    # ---- Identify where ACI adds value ----
    print("\n" + "="*70)
    print("ANALYSIS: Where Does ACI-DQN Add Value?")
    print("="*70)
    for scenario in full_df['scenario'].unique():
        sdf = full_df[full_df['scenario'] == scenario]
        print(f"\n  {scenario}:")
        dqn_row = sdf[sdf['method'] == 'dqn']
        aci_row = sdf[sdf['method'] == 'aci_dqn']
        greedy_row = sdf[sdf['method'] == 'queue_greedy']

        if len(greedy_row) > 0:
            g = greedy_row.iloc[0]
            print(f"  Greedy baseline:            cost={g['total_cost']:.0f}, "
                  f"P3_viol={int(g['P3_violations'])}, "
                  f"P3_comp={int(g['P3_completed'])}/{int(g['total_P3'])}")
        if len(dqn_row) > 0 and len(aci_row) > 0:
            d = dqn_row.iloc[0]
            a = aci_row.iloc[0]
            print(f"  DQN    -> ACI-DQN:          cost {d['total_cost']:.0f} -> {a['total_cost']:.0f} "
                  f"({(a['total_cost']-d['total_cost'])/d['total_cost']*100:+.1f}%), "
                  f"P3_viol {int(d['P3_violations'])} -> {int(a['P3_violations'])}")

    print(f"\nFull results -> {OUT_DIR / 'stress_scenario_results.csv'}")
    print("Done.")
