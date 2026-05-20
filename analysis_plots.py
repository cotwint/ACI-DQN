"""
analysis_plots.py
-----------------
Generate comprehensive analysis plots and stress test for the
data center computing-power co-scheduling experiment.

Usage:
    python analysis_plots.py
"""

import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

from src.utils import load_config, set_global_seed, ensure_dir
from src.data_preprocess import load_processed, day_matrix, normalised_day_matrix
from src.datacenter_env import DataCenterEnv, action_to_n, n_to_action
from _common import build_env_and_splits
from src.rl.dqn_agent import DQNAgent
from src.rl.train_dqn import rollout_episode, IdentityAugmenter, EpisodeStats
from src.rl.augmenters import ConformalAugmenter
from src.evaluation.metrics import episode_stats_to_row
from src.baselines.fixed_policy import FixedPolicy
from src.baselines.queue_greedy_policy import QueueGreedyPolicy
from src.baselines.price_aware_greedy_policy import PriceAwareGreedyPolicy
from _heuristic_runner import run_heuristic

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
FIG_DIR = ensure_dir("outputs/figures")
cfg = load_config("config.yaml")
set_global_seed(2024)

env, splits, norm = build_env_and_splits(cfg)
daily_df = pd.read_csv("outputs/daily_results.csv")

# 9-method palette
C9 = {
    'fixed': '#607D8B', 'queue_greedy': '#4CAF50',
    'price_aware_greedy': '#8BC34A', 'forecast_greedy': '#00BCD4',
    'conformal_greedy': '#FF5722', 'dqn': '#2196F3',
    'forecast_dqn': '#3F51B5', 'static_conformal_dqn': '#795548',
    'aci_dqn': '#9C27B0',
}
ORDER9 = ['fixed', 'queue_greedy', 'price_aware_greedy',
          'forecast_greedy', 'conformal_greedy',
          'dqn', 'forecast_dqn', 'static_conformal_dqn', 'aci_dqn']
LABELS9 = ['Fixed', 'Queue-\nGreedy', 'Price-Aware\nGreedy',
           'Forecast-\nGreedy', 'Conformal-\nGreedy',
           'DQN', 'Forecast-\nDQN', 'Static-C.\nDQN', 'ACI-DQN']

# =========================================================================
# Figure 1: Cost decomposition bar chart
# =========================================================================
def fig1_cost_decomposition():
    summary = pd.read_csv("outputs/experiment_summary.csv")
    order = ORDER9
    labels = LABELS9

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Cost breakdown
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.25
    elec = [summary[summary['method']==m]['electricity_cost'].values[0] for m in order]
    qos = [summary[summary['method']==m]['qos_cost'].values[0] for m in order]
    switch = [summary[summary['method']==m]['switching_cost'].values[0] for m in order]
    ax.bar(x - w, elec, w, label='Electricity', color='#2196F3')
    ax.bar(x, qos, w, label='QoS Penalty', color='#F44336')
    ax.bar(x + w, switch, w, label='Switching', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Cost (CNY/day)')
    ax.set_title('(a) Cost Decomposition by Method')
    ax.legend(fontsize=8)

    # Server utilization
    ax = axes[1]
    servers = [summary[summary['method']==m]['average_active_servers'].values[0] for m in order]
    utils = [summary[summary['method']==m]['average_utilization'].values[0] for m in order]
    colors = [C9[m] for m in order]
    ax2 = ax.twinx()
    bars = ax.bar(x, servers, 0.5, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Avg Active Servers', color='#333')
    ax2.plot(x, utils, 'ko-', linewidth=2, markersize=8)
    ax2.set_ylabel('Avg Utilization', color='#333')
    for i, (s, u) in enumerate(zip(servers, utils)):
        ax.text(i, s + 1, f'{s:.1f}', ha='center', fontsize=7)
        ax2.annotate(f'{u:.3f}', (i, u), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=7)
    ax.set_title('(b) Server Count & Utilization')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cost_decomposition.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] fig1_cost_decomposition.png")


# =========================================================================
# Figure 2: Per-priority SLA and delay
# =========================================================================
def fig2_sla_breakdown():
    summary = pd.read_csv("outputs/experiment_summary.csv")
    order = ORDER9
    labels = ['Fixed', 'Queue-Greedy', 'Price-Greedy', 'Forecast-Greedy',
              'Conformal-Greedy', 'DQN', 'Forecast-DQN', 'Static-C-DQN', 'ACI-DQN']

    colors_sla = [C9[m] for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    priorities = ['P1', 'P2', 'P3']

    for pi, (p, ax) in enumerate(zip(priorities, axes)):
        violation_rates = [summary[summary['method']==m][f'{p}_sla_violation_rate'].values[0] for m in order]
        avg_delays = [summary[summary['method']==m][f'{p}_avg_delay'].values[0] for m in order]
        completed = [summary[summary['method']==m][f'{p}_completed'].values[0] for m in order]

        x = np.arange(len(order))
        ax2 = ax.twinx()
        bars = ax.bar(x, avg_delays, 0.4, color=colors_sla, alpha=0.7)
        ax2.plot(x, violation_rates, 'ro-', linewidth=2, markersize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Avg Delay (slots)', fontsize=10)
        ax2.set_ylabel('Violation Rate', color='red', fontsize=10)
        ax.set_title(f'{p}: Delay & Violation Rate', fontsize=11)

        for i, (d, v, c) in enumerate(zip(avg_delays, violation_rates, completed)):
            ax.text(i, d + 0.3, f'{d:.1f}', ha='center', fontsize=7)
            ax2.annotate(f'{v:.4f}', (i, v), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=7, color='red')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_sla_breakdown.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] fig2_sla_breakdown.png")


# =========================================================================
# Figure 3: Action distribution and dynamics
# =========================================================================
def fig3_action_dynamics():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # 3a: Daily server variation boxplot
    ax = axes[0]
    order = ORDER9
    labels = ['Fixed', 'Queue-\nGreedy', 'Price-\nGreedy', 'Forecast-\nGreedy',
              'Conf.\nGreedy', 'DQN', 'Forecast-\nDQN', 'Static-C.\nDQN', 'ACI-\nDQN']
    box_data = [daily_df[daily_df['method']==m]['average_active_servers'].values for m in order]
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    colors_box = [C9[m] for m in order]
    for patch, c in zip(bp['boxes'], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Avg Active Servers')
    ax.set_title('(a) Server Count Distribution across 267 Days')
    ax.tick_params(axis='x', labelsize=7)

    # 3b: Training reward curves
    ax = axes[1]
    rl_methods = ['dqn', 'forecast_dqn', 'static_conformal_dqn', 'aci_dqn']
    rl_labels = ['DQN', 'Forecast-DQN', 'Static-C-DQN', 'ACI-DQN']
    rl_colors = [C9[m] for m in rl_methods]
    for m, label, c in zip(rl_methods, rl_labels, rl_colors):
        try:
            hist = pd.read_csv(f'outputs/{m}_training_history.csv')
            # Smooth with rolling mean
            smoothed = hist['reward'].rolling(20, min_periods=1).mean()
            ax.plot(smoothed.values, color=c, label=label, alpha=0.8, linewidth=1.5)
        except Exception:
            pass
    ax.set_xlabel('Episode')
    ax.set_ylabel('Smoothed Reward (window=20)')
    ax.set_title('(b) Training Reward Curves')
    ax.legend(fontsize=8)

    # 3c: Training loss curves
    ax = axes[2]
    for m, label, c in zip(rl_methods, rl_labels, rl_colors):
        try:
            hist = pd.read_csv(f'outputs/{m}_training_history.csv')
            smoothed = hist['avg_loss'].rolling(20, min_periods=1).mean()
            ax.plot(smoothed.values, color=c, label=label, alpha=0.8, linewidth=1.5)
        except Exception:
            pass
    ax.set_xlabel('Episode')
    ax.set_ylabel('Smoothed Avg Loss (window=20)')
    ax.set_title('(c) Training Loss Curves')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_action_dynamics.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] fig3_action_dynamics.png")


# =========================================================================
# Figure 4: Conformal prediction diagnostics
# =========================================================================
def fig4_conformal_diagnostics():
    """Run per-slot analysis on a sample day for ACI coverage diagnostics."""
    try:
        agent_aci = DQNAgent(
            state_dim=env.observation_dim + 2 * cfg['qos']['K'],
            action_dim=env.action_dim,
            hidden=cfg['rl']['hidden_sizes'], lr=cfg['rl']['lr'],
            batch_size=cfg['rl']['batch_size'], replay_size=cfg['rl']['replay_size'],
            epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.995,
            target_update_interval=cfg['rl']['target_update_interval'],
            learning_starts=cfg['rl']['learning_starts'],
            reward_scale=cfg['rl']['reward_scale'],
            max_grad_norm=cfg['rl']['max_grad_norm'],
        )
        agent_aci.load('outputs/models/aci_dqn.pt')
        agent_aci.epsilon = 0.0
    except Exception:
        print("  [SKIP] fig4: models not found, run experiment first")
        return

    aug_aci = ConformalAugmenter(cfg, learner='aci', use_shield=False)
    aug_aci.reset(env, 1064)

    # Collect per-slot conformal intervals for a sample day
    test_day = 1200
    env_aci = DataCenterEnv(cfg=cfg, day_load_matrix=env.day_load_matrix,
                            day_norm_matrix=env.day_norm_matrix, base_seed=2024)

    state_aci, _ = env_aci.reset(test_day)

    slots_data = []
    for slot in range(96):
        s_aug = aug_aci.augment(state_aci, env_aci)
        lo, hi = aug_aci.cp.intervals_h_steps()
        a_raw = agent_aci.select_action(s_aug, greedy=True)
        a_n = action_to_n(a_raw, cfg)
        a_safe = aug_aci.shield(a_n, s_aug, env_aci)
        a_safe_idx = n_to_action(a_safe, cfg)
        ns_aci, r, d, info_aci = env_aci.step(a_safe_idx)
        actual_p3 = info_aci['arrivals'][2]
        aug_aci.on_step(env_aci, info_aci, a_raw, a_safe)

        slots_data.append({
            'slot': slot,
            'actual_p3': actual_p3,
            'aci_hi_p3': hi[2, 0],
            'aci_lo_p3': lo[2, 0],
        })
        state_aci = ns_aci
        if d: break

    df_s = pd.DataFrame(slots_data)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(df_s['slot'], df_s['aci_lo_p3'], df_s['aci_hi_p3'],
                    alpha=0.3, color='#9C27B0', label='ACI interval')
    ax.plot(df_s['slot'], df_s['actual_p3'], 'o-', color='#333', markersize=3, label='Actual P3 arrivals')
    aci_covered = ((df_s['actual_p3'] >= df_s['aci_lo_p3']) & (df_s['actual_p3'] <= df_s['aci_hi_p3'])).mean()
    ax.set_title(f'ACI Conformal Intervals — P3 (coverage={aci_covered:.2%})')
    ax.set_xlabel('Slot (15-min interval)')
    ax.set_ylabel('P3 Arrivals per Slot')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_conformal_diagnostics.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] fig4_conformal_diagnostics.png (ACI cov={aci_covered:.2%})")


# =========================================================================
# Figure 5: DQN action sensitivity analysis
# =========================================================================
def fig5_dqn_sensitivity():
    """Analyze DQN's action response to P3 queue state."""
    try:
        agent_dqn = DQNAgent(
            state_dim=env.observation_dim, action_dim=env.action_dim,
            hidden=cfg['rl']['hidden_sizes'], lr=cfg['rl']['lr'],
            batch_size=cfg['rl']['batch_size'], replay_size=cfg['rl']['replay_size'],
            epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.995,
            target_update_interval=cfg['rl']['target_update_interval'],
            learning_starts=cfg['rl']['learning_starts'],
            reward_scale=cfg['rl']['reward_scale'],
            max_grad_norm=cfg['rl']['max_grad_norm'],
        )
        agent_dqn.load('outputs/models/dqn.pt')
        agent_dqn.epsilon = 0.0
    except Exception:
        print("  [SKIP] fig5: DQN model not found")
        return

    aug = IdentityAugmenter()
    env_test = DataCenterEnv(cfg=cfg, day_load_matrix=env.day_load_matrix,
                             day_norm_matrix=env.day_norm_matrix, base_seed=2024)

    # Collect per-slot data across several days
    all_slots = []
    for day in [1064, 1100, 1150, 1200, 1250, 1300]:
        state, _ = env_test.reset(day)
        for slot in range(96):
            a_raw = agent_dqn.select_action(state, greedy=True)
            a_n = action_to_n(a_raw, cfg)
            a_safe = aug.shield(a_n, state, env_test)
            a_safe_idx = n_to_action(a_safe, cfg)
            next_state, r, d, info = env_test.step(a_safe_idx)

            # Extract state components (de-normalize)
            q_len = info['queue_len']
            bl = info['backlog_work']
            price_norm = state[12]

            all_slots.append({
                'day': day, 'slot': slot,
                'action': a_raw,
                'n_servers': a_safe,
                'util': info['util'],
                'q1_len': q_len[0], 'q2_len': q_len[1], 'q3_len': q_len[2],
                'b1': bl[0], 'b2': bl[1], 'b3': bl[2],
                'price_norm': price_norm,
            })
            state = next_state
            if d: break

    df_s = pd.DataFrame(all_slots)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 5a: n_servers vs P3 queue length
    ax = axes[0, 0]
    ax.scatter(df_s['q3_len'], df_s['n_servers'], c=df_s['price_norm'], cmap='coolwarm', alpha=0.5, s=20)
    ax.set_xlabel('P3 Queue Length')
    ax.set_ylabel('Active Servers')
    ax.set_title(f'(a) Servers vs P3 Queue (r={df_s["n_servers"].corr(df_s["q3_len"]):.3f})')
    plt.colorbar(ax.collections[0], ax=ax, label='Price (norm)')

    # 5b: n_servers vs P3 backlog
    ax = axes[0, 1]
    ax.scatter(df_s['b3'], df_s['n_servers'], c=df_s['price_norm'], cmap='coolwarm', alpha=0.5, s=20)
    ax.set_xlabel('P3 Backlog Work')
    ax.set_ylabel('Active Servers')
    ax.set_title(f'(b) Servers vs P3 Backlog (r={df_s["n_servers"].corr(df_s["b3"]):.3f})')
    plt.colorbar(ax.collections[0], ax=ax, label='Price (norm)')

    # 5c: Utilization histogram
    ax = axes[1, 0]
    ax.hist(df_s['util'], bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(df_s['util'].mean(), color='red', linestyle='--', label=f'Mean={df_s["util"].mean():.3f}')
    ax.set_xlabel('Utilization')
    ax.set_ylabel('Frequency')
    ax.set_title('(c) Utilization Distribution')
    ax.legend()

    # 5d: Action histogram
    ax = axes[1, 1]
    action_counts = df_s['action'].value_counts().sort_index()
    ax.bar(action_counts.index, action_counts.values, color='#2196F3', alpha=0.7)
    ax.set_xlabel('Action Index (0-20)')
    ax.set_ylabel('Count')
    ax.set_title(f'(d) Action Distribution (6 days, {len(df_s)} slots)')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_dqn_sensitivity.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] fig5_dqn_sensitivity.png")


# =========================================================================
# Stress Test: High Price + P3 Surge + Deadline Concentration
# =========================================================================
def generate_stress_test():
    """Generate a synthetic stress test day and run all baselines."""
    print("\n=== Stress Test Generation ===")

    from src.workload_generator import Task, DayWorkload

    T = 96
    K = 3
    deadlines = np.array(cfg['qos']['deadline_slots'])
    service_mean = np.array(cfg['qos']['service_mean'])
    service_std = np.array(cfg['qos']['service_std'])

    # Modify price curve: 3x during slots 40-70
    base_price = env.price_curve.copy()
    surge_slots = np.arange(40, 71)
    base_price[surge_slots] *= 3.0

    # Generate task trace with P3 surge + deadline concentration
    rng = np.random.default_rng(9999)
    tasks_per_slot = [[] for _ in range(T) for __ in range(K)]
    tasks_per_slot = [[[] for _ in range(T)] for _ in range(K)]
    arrival_work = np.zeros((K, T), dtype=np.float64)

    total_tasks = 0
    total_p3 = 0
    p3_surge_count = 0

    for t in range(T):
        p3_rate = 7.2 if 30 <= t <= 70 else 2.4
        arrival_rates = [4.2, 4.6, p3_rate]

        for k in range(K):
            n_arrive = rng.poisson(arrival_rates[k])
            if n_arrive == 0:
                continue
            # Generate work using gamma (shape=mean, scale=0.3) clamped
            work_vals = rng.normal(loc=service_mean[k], scale=service_std[k], size=n_arrive)
            work_vals = np.clip(work_vals, 0.05, None)

            for j in range(n_arrive):
                # P3 shorter deadlines during surge window
                if k == 2 and 30 <= t <= 70:
                    d_slot = t + int(rng.integers(1, max(2, int(deadlines[k] * 0.3)) + 1))
                    p3_surge_count += 1
                else:
                    d_slot = t + int(deadlines[k])
                d_slot = min(T - 1, max(t + 1, d_slot))

                tasks_per_slot[k][t].append(
                    Task(arrival=t, deadline=d_slot,
                         work=float(work_vals[j]), remaining=float(work_vals[j]))
                )
                total_tasks += 1
                if k == 2:
                    total_p3 += 1
            arrival_work[k, t] = float(work_vals.sum())

    print(f"  Generated {total_tasks} tasks ({total_p3} P3, {p3_surge_count} in surge window)")

    # Helper to create a fresh (deep-copied) workload for each policy run
    def fresh_workload():
        fresh_tsp = [[[] for _ in range(T)] for _ in range(K)]
        for k in range(K):
            for t in range(T):
                for task in tasks_per_slot[k][t]:
                    fresh_tsp[k][t].append(Task(
                        arrival=task.arrival, deadline=task.deadline,
                        work=task.work, remaining=task.work))
        return DayWorkload(
            lam=np.zeros((K, T)),
            n_arrivals=np.array([[len(fresh_tsp[k][t]) for t in range(T)] for k in range(K)]),
            arrival_work=arrival_work.copy(),
            tasks_per_slot=fresh_tsp,
        )

    # Run heuristics using a modified env
    def make_stress_env():
        e = DataCenterEnv(cfg=cfg, day_load_matrix=env.day_load_matrix,
                           day_norm_matrix=env.day_norm_matrix, base_seed=2024)
        return e

    methods_cfg = {
        'fixed': ('Fixed', FixedPolicy(cfg)),
        'queue_greedy': ('Queue-Greedy', QueueGreedyPolicy(cfg)),
        'price_aware_greedy': ('Price-Greedy', PriceAwareGreedyPolicy(cfg)),
    }

    stress_results = []

    for mkey, (mlabel, policy) in methods_cfg.items():
        env_s = make_stress_env()
        # Override reset to inject stress workload and price
        state, _ = env_s.reset(0)
        env_s.workload = fresh_workload()
        env_s.set_price_curve(base_price)
        policy.reset()

        for slot in range(T):
            n = policy.act(state, env_s)
            a_idx = n_to_action(n, cfg)
            next_state, r, d, info = env_s.step(a_idx)
            state = next_state
            if d:
                break

        h = env_s.history_as_arrays()
        stress_results.append({
            'method': mkey,
            'label': mlabel,
            'total_cost': float(h['total_cost'].sum()),
            'elec_cost': float(h['elec_cost'].sum()),
            'qos_cost': float(h['qos_cost'].sum()),
            'avg_servers': float(h['n_active'].mean()),
            'avg_util': float(h['util'].mean()),
            'P1_violations': float(h['violations'].sum(axis=0)[0]),
            'P2_violations': float(h['violations'].sum(axis=0)[1]),
            'P3_violations': float(h['violations'].sum(axis=0)[2]),
            'P3_completed': float(h['completed'].sum(axis=0)[2]),
            'total_P3': float(h['arrivals'].sum(axis=0)[2]),
        })

    # Run RL methods
    try:
        rl_configs = [
            ('dqn', 0),
            ('forecast_dqn', cfg['qos']['K']),
            ('static_conformal_dqn', 2 * cfg['qos']['K']),
            ('aci_dqn', 2 * cfg['qos']['K']),
        ]
        for mkey, extra_dim in rl_configs:
            agent = DQNAgent(
                state_dim=env.observation_dim + extra_dim,
                action_dim=env.action_dim,
                hidden=cfg['rl']['hidden_sizes'], lr=cfg['rl']['lr'],
                batch_size=cfg['rl']['batch_size'], replay_size=cfg['rl']['replay_size'],
                epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.995,
                target_update_interval=cfg['rl']['target_update_interval'],
                learning_starts=cfg['rl']['learning_starts'],
                reward_scale=cfg['rl']['reward_scale'],
                max_grad_norm=cfg['rl']['max_grad_norm'],
            )
            agent.load(f'outputs/models/{mkey}.pt')
            agent.epsilon = 0.0

            env_s = make_stress_env()
            state, _ = env_s.reset(0)
            env_s.workload = fresh_workload()
            env_s.set_price_curve(base_price)

            if mkey == 'dqn':
                aug = IdentityAugmenter()
            elif mkey == 'forecast_dqn':
                from src.rl.augmenters import ForecastAugmenter
                aug = ForecastAugmenter(cfg)
            elif mkey == 'static_conformal_dqn':
                from src.rl.augmenters import StaticConformalAugmenter
                aug = StaticConformalAugmenter(cfg)
            else:
                aug = ConformalAugmenter(cfg, learner='aci', use_shield=False)

            aug.reset(env_s, 0)
            for slot in range(T):
                s_aug = aug.augment(state, env_s)
                a_raw = agent.select_action(s_aug, greedy=True)
                a_n = action_to_n(a_raw, cfg)
                a_safe = aug.shield(a_n, s_aug, env_s)
                a_safe_idx = n_to_action(a_safe, cfg)
                next_state, r, d, info = env_s.step(a_safe_idx)
                aug.on_step(env_s, info, a_raw, a_safe)
                state = next_state
                if d:
                    break

            h = env_s.history_as_arrays()
            stress_results.append({
                'method': mkey,
                'label': mkey.upper(),
                'total_cost': float(h['total_cost'].sum()),
                'elec_cost': float(h['elec_cost'].sum()),
                'qos_cost': float(h['qos_cost'].sum()),
                'avg_servers': float(h['n_active'].mean()),
                'avg_util': float(h['util'].mean()),
                'P1_violations': float(h['violations'].sum(axis=0)[0]),
                'P2_violations': float(h['violations'].sum(axis=0)[1]),
                'P3_violations': float(h['violations'].sum(axis=0)[2]),
                'P3_completed': float(h['completed'].sum(axis=0)[2]),
                'total_P3': float(h['arrivals'].sum(axis=0)[2]),
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [WARN] Could not run RL on stress test: {e}")

    stress_df = pd.DataFrame(stress_results)
    stress_path = FIG_DIR.parent / "stress_test_results.csv"
    stress_df.to_csv(stress_path, index=False)
    print(f"  Stress test results -> {stress_path}")

    # Stress test figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    order = ORDER9
    colors_stress = [C9[m] for m in order]
    labels_stress = ['Fixed', 'Queue-\nGreedy', 'Price-\nGreedy', 'Forecast-\nGreedy',
                     'Conf.\nGreedy', 'DQN', 'Forecast-\nDQN', 'Static-C.\nDQN', 'ACI-\nDQN']

    # Cost
    ax = axes[0]
    lookup = dict(zip(stress_df['method'], stress_df['total_cost']))
    costs = [lookup.get(m, 0) for m in order]
    ax.bar(range(len(order)), costs, color=colors_stress)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels_stress, fontsize=8)
    ax.set_ylabel('Total Cost (CNY)')
    ax.set_title('(a) Stress Day Total Cost')

    # P3 completed
    ax = axes[1]
    lookup_c = dict(zip(stress_df['method'], stress_df['P3_completed']))
    lookup_t = dict(zip(stress_df['method'], stress_df['total_P3']))
    p3_comp = [lookup_c.get(m, 0) for m in order]
    p3_total = max([lookup_t.get(m, 0) for m in order], default=0)
    ax.bar(range(len(order)), p3_comp, color=colors_stress, alpha=0.7)
    ax.axhline(y=p3_total, color='red', linestyle='--', linewidth=1.5,
               label=f'Total P3={p3_total:.0f}')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels_stress, fontsize=8)
    ax.set_ylabel('P3 Completed')
    ax.set_title('(b) P3 Task Completion')
    ax.legend(fontsize=7)

    # P3 violations
    ax = axes[2]
    lookup_v = dict(zip(stress_df['method'], stress_df['P3_violations']))
    p3_viol = [lookup_v.get(m, 0) for m in order]
    ax.bar(range(len(order)), p3_viol,
           color=['#F44336' if v > 0 else '#4CAF50' for v in p3_viol])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels_stress, fontsize=8)
    ax.set_ylabel('P3 SLA Violations')
    ax.set_title('(c) P3 SLA Violations on Stress Day')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_stress_test.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] fig6_stress_test.png")

    return stress_df


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("Generating analysis plots...")
    fig1_cost_decomposition()
    fig2_sla_breakdown()
    fig3_action_dynamics()
    fig4_conformal_diagnostics()
    fig5_dqn_sensitivity()
    stress_df = generate_stress_test()
    print(f"\nAll figures saved to {FIG_DIR}/")
    print(f"Stress test results: outputs/stress_test_results.csv")
