"""
generate_figures.py
------------------
Generate publication-quality figures for the data center
computing-power co-scheduling experiment.

Produces 5 figures in outputs/figures/:
  fig1_cost_efficiency.png  – cost decomposition + server/util comparison
  fig2_training_dynamics.png – reward & loss curves
  fig3_stress_test.png       – S2 burst scenario (key paper result)
  fig4_conformal.png         – conformal interval diagnostics
  fig5_action_policy.png     – per-method action distributions
"""

import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 11,
})

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
C = {
    'fixed': '#607D8B', 'queue_greedy': '#4CAF50',
    'price_aware_greedy': '#8BC34A', 'dqn': '#2196F3',
    'aci_dqn': '#9C27B0', 'dtaci_dqn': '#E91E63',
}
ORDER = ['fixed', 'queue_greedy', 'price_aware_greedy', 'dqn', 'aci_dqn', 'dtaci_dqn']
LABELS = ['Fixed', 'Queue-\nGreedy', 'Price-\nGreedy', 'DQN', 'ACI-DQN', 'DtACI-DQN']
LABELS_SHORT = ['Fixed', 'Queue-Greedy', 'Price-Greedy', 'DQN', 'ACI-DQN', 'DtACI-DQN']

summary = pd.read_csv("outputs/experiment_summary.csv")
daily = pd.read_csv("outputs/daily_results.csv")

# ==============================================================================
# Figure 1: Cost Decomposition + Server/Util Efficiency
# ==============================================================================
def fig1_cost_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) Cost decomposition
    ax = axes[0]
    x = np.arange(len(ORDER))
    w = 0.25
    elec = [summary[summary['method'] == m]['electricity_cost'].values[0] for m in ORDER]
    qos = [summary[summary['method'] == m]['qos_cost'].values[0] for m in ORDER]
    switch = [summary[summary['method'] == m]['switching_cost'].values[0] for m in ORDER]
    ax.bar(x - w, elec, w, label='Electricity Cost', color='#64B5F6', edgecolor='white', linewidth=0.5)
    ax.bar(x, qos, w, label='QoS Penalty', color='#EF5350', edgecolor='white', linewidth=0.5)
    ax.bar(x + w, switch, w, label='Switching Cost', color='#FFB74D', edgecolor='white', linewidth=0.5)
    for i, (e, q, s) in enumerate(zip(elec, qos, switch)):
        total = e + q + s
        ax.text(i, total + 5, f'{total:.0f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.set_ylabel('Daily Cost (CNY)', fontsize=11)
    ax.set_title('(a) Cost Decomposition by Method', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, max(elec) * 1.25)

    # (b) Server count + utilization dual-axis
    ax = axes[1]
    servers = [summary[summary['method'] == m]['average_active_servers'].values[0] for m in ORDER]
    utils = [summary[summary['method'] == m]['average_utilization'].values[0] for m in ORDER]
    colors = [C[m] for m in ORDER]
    ax2 = ax.twinx()
    bars = ax.bar(x, servers, 0.5, color=colors, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.set_ylabel('Avg Active Servers', fontsize=11, color='#333')
    ax.set_ylim(0, max(servers) * 1.2)
    ax2.plot(x, utils, 'ko-', linewidth=2.2, markersize=9, markerfacecolor='white', markeredgewidth=1.5)
    ax2.set_ylabel('Avg Utilization', fontsize=11, color='#333')
    ax2.set_ylim(0, 1.15)
    ax2.axhline(y=1.0, color='#333', linestyle=':', linewidth=0.8, alpha=0.5)
    for i, (s, u) in enumerate(zip(servers, utils)):
        ax.text(i, s + 1.2, f'{s:.1f}', ha='center', fontsize=8, fontweight='bold')
        ax2.annotate(f'{u:.1%}', (i, u), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, fontweight='bold')
    ax.set_title('(b) Server Count & Utilization', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cost_efficiency.png")
    plt.close()
    print("  [OK] fig1_cost_efficiency.png")


# ==============================================================================
# Figure 2: Training Dynamics
# ==============================================================================
def fig2_training_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rl_methods = ['dqn', 'aci_dqn', 'dtaci_dqn']
    rl_labels = ['DQN', 'ACI-DQN', 'DtACI-DQN']
    rl_colors = [C[m] for m in rl_methods]
    window = 25

    # (a) Reward
    ax = axes[0]
    for m, label, color in zip(rl_methods, rl_labels, rl_colors):
        try:
            hist = pd.read_csv(f'outputs/{m}_training_history.csv')
            smoothed = hist['reward'].rolling(window, min_periods=1, center=True).mean()
            episodes = np.arange(len(smoothed))
            ax.plot(episodes, smoothed, color=color, label=label, linewidth=1.6, alpha=0.9)
            # Shade std
            std = hist['reward'].rolling(window, min_periods=1, center=True).std()
            ax.fill_between(episodes, smoothed - std * 0.3, smoothed + std * 0.3,
                            color=color, alpha=0.1)
        except Exception:
            pass
    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Episode Reward (smoothed)', fontsize=11)
    ax.set_title(f'(a) Training Reward (window={window})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Loss
    ax = axes[1]
    for m, label, color in zip(rl_methods, rl_labels, rl_colors):
        try:
            hist = pd.read_csv(f'outputs/{m}_training_history.csv')
            smoothed = hist['avg_loss'].rolling(window, min_periods=1, center=True).mean()
            episodes = np.arange(len(smoothed))
            ax.plot(episodes, smoothed, color=color, label=label, linewidth=1.6, alpha=0.9)
        except Exception:
            pass
    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Avg TD Loss (smoothed)', fontsize=11)
    ax.set_title(f'(b) Training Loss (window={window})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_training_dynamics.png")
    plt.close()
    print("  [OK] fig2_training_dynamics.png")


# ==============================================================================
# Figure 3: S1 & S2 Performance Comparison (comprehensive)
# ==============================================================================
def fig3_stress_test():
    stress = pd.read_csv("outputs/stress_test/stress_scenario_results.csv")

    s1_df = stress[stress['scenario'].str.startswith('S1_')]
    s2_df = stress[stress['scenario'].str.startswith('S2_')]

    fig = plt.figure(figsize=(18, 11))

    def get_vals(df, col):
        result = []
        for m in ORDER:
            row = df[df['method'] == m]
            result.append(row[col].values[0] if len(row) > 0 else 0)
        return result

    # ============================================================
    # Row 1: S1 — Seasonal Peak Shift (Distribution Shift)
    # ============================================================
    # (a) S1 Total Cost
    ax = fig.add_subplot(2, 4, 1)
    costs_s1 = get_vals(s1_df, 'total_cost')
    bars = ax.bar(range(len(ORDER)), costs_s1, color=[C[m] for m in ORDER],
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Total Cost (CNY)', fontsize=10)
    ax.set_title('(a) S1 Total Cost', fontsize=11, fontweight='bold')
    for i, c in enumerate(costs_s1):
        color = '#9C27B0' if ORDER[i] == 'aci_dqn' else '#333'
        ax.text(i, c + max(costs_s1) * 0.015, f'{c:.1f}', ha='center', fontsize=7.5,
                fontweight='bold' if ORDER[i] == 'aci_dqn' else 'normal', color=color)
    # Highlight best
    best_idx = np.argmin(costs_s1)
    bars[best_idx].set_edgecolor('#FFD700')
    bars[best_idx].set_linewidth(2.5)

    # (b) S1 Server Count
    ax = fig.add_subplot(2, 4, 2)
    servers_s1 = get_vals(s1_df, 'avg_servers')
    bars = ax.bar(range(len(ORDER)), servers_s1, color=[C[m] for m in ORDER],
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Avg Servers', fontsize=10)
    ax.set_title('(b) S1 Server Count', fontsize=11, fontweight='bold')
    for i, s in enumerate(servers_s1):
        ax.text(i, s + max(servers_s1) * 0.015, f'{s:.1f}', ha='center', fontsize=7.5,
                fontweight='bold' if ORDER[i] == 'aci_dqn' else 'normal')

    # (c) S1 Utilization
    ax = fig.add_subplot(2, 4, 3)
    utils_s1 = get_vals(s1_df, 'avg_util')
    util_colors = []
    for i, u in enumerate(utils_s1):
        if u >= 0.99:
            util_colors.append('#F44336')  # red: dangerously high
        elif u < 0.50:
            util_colors.append('#FF9800')  # orange: too low (waste)
        else:
            util_colors.append('#4CAF50')  # green: healthy range
    bars = ax.bar(range(len(ORDER)), [u * 100 for u in utils_s1], color=util_colors,
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Utilization (%)', fontsize=10)
    ax.set_title('(c) S1 Utilization', fontsize=11, fontweight='bold')
    ax.axhline(y=100, color='#F44336', linestyle='--', alpha=0.5, linewidth=0.8, label='100% (no buffer)')
    ax.axhline(y=50, color='#FF9800', linestyle='--', alpha=0.5, linewidth=0.8, label='50% (over-provisioned)')
    for i, u in enumerate(utils_s1):
        ax.text(i, u * 100 + 2, f'{u:.1%}', ha='center', fontsize=7.5,
                fontweight='bold' if ORDER[i] == 'aci_dqn' else 'normal')
    ax.legend(fontsize=6.5, loc='upper right')

    # (d) S1 Cost Breakdown (stacked bars)
    ax = fig.add_subplot(2, 4, 4)
    elec_s1 = get_vals(s1_df, 'elec_cost')
    qos_s1 = get_vals(s1_df, 'qos_cost')
    switch_s1 = get_vals(s1_df, 'switch_cost')
    x = np.arange(len(ORDER))
    w = 0.6
    ax.bar(x, elec_s1, w, label='Electricity', color='#64B5F6', edgecolor='white', linewidth=0.5)
    ax.bar(x, qos_s1, w, bottom=elec_s1, label='QoS Penalty', color='#EF5350', edgecolor='white', linewidth=0.5)
    bottom2 = [e + q for e, q in zip(elec_s1, qos_s1)]
    ax.bar(x, switch_s1, w, bottom=bottom2, label='Switching', color='#FFB74D', edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Cost (CNY)', fontsize=10)
    ax.set_title('(d) S1 Cost Breakdown', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    for i, (e, q, s) in enumerate(zip(elec_s1, qos_s1, switch_s1)):
        total = e + q + s
        ax.text(i, total + max(costs_s1) * 0.015, f'{total:.0f}', ha='center', fontsize=7.5, fontweight='bold')

    # ============================================================
    # Row 2: S2 — Burst Anomaly (High Randomness)
    # ============================================================
    # (e) S2 Total Cost
    ax = fig.add_subplot(2, 4, 5)
    costs_s2 = get_vals(s2_df, 'total_cost')
    bars = ax.bar(range(len(ORDER)), costs_s2, color=[C[m] for m in ORDER],
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Total Cost (CNY)', fontsize=10)
    ax.set_title('(e) S2 Total Cost', fontsize=11, fontweight='bold')
    for i, c in enumerate(costs_s2):
        color = '#9C27B0' if ORDER[i] == 'aci_dqn' else '#333'
        ax.text(i, c + max(costs_s2) * 0.015, f'{c:.1f}', ha='center', fontsize=7.5,
                fontweight='bold' if ORDER[i] == 'aci_dqn' else 'normal', color=color)
    best_idx = np.argmin(costs_s2)
    bars[best_idx].set_edgecolor('#FFD700')
    bars[best_idx].set_linewidth(2.5)

    # (f) S2 P3 SLA Violations (the key differentiator!)
    ax = fig.add_subplot(2, 4, 6)
    viols_s2 = [int(v) for v in get_vals(s2_df, 'P3_violations')]
    p3_total = int(s2_df['total_P3'].values[0])
    viol_colors = ['#F44336' if v > 0 else '#2E7D32' for v in viols_s2]
    bars = ax.bar(range(len(ORDER)), viols_s2, color=viol_colors, edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('P3 SLA Violations', fontsize=10)
    ax.set_title(f'(f) S2 P3 Violations (total={p3_total})', fontsize=11, fontweight='bold')
    for i, v in enumerate(viols_s2):
        y_pos = v + max(1, max(viols_s2) * 0.05)
        badge = ' FAIL' if v > 0 else ' PASS'
        ax.text(i, y_pos, f'{v}{badge}', ha='center', fontsize=7,
                fontweight='bold', color='#C62828' if v > 0 else '#1B5E20')
    # Add zero-line emphasis
    ax.axhline(y=0, color='#333', linewidth=0.8)

    # (g) S2 Server Count
    ax = fig.add_subplot(2, 4, 7)
    servers_s2 = get_vals(s2_df, 'avg_servers')
    bars = ax.bar(range(len(ORDER)), servers_s2, color=[C[m] for m in ORDER],
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(LABELS, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Avg Servers', fontsize=10)
    ax.set_title('(g) S2 Server Count', fontsize=11, fontweight='bold')
    for i, s in enumerate(servers_s2):
        ax.text(i, s + max(servers_s2) * 0.015, f'{s:.1f}', ha='center', fontsize=7.5,
                fontweight='bold' if ORDER[i] == 'aci_dqn' else 'normal')

    # (h) S2 Cost vs Violations Scatter
    ax = fig.add_subplot(2, 4, 8)
    for m in ORDER:
        row_c = costs_s2[ORDER.index(m)]
        row_v = viols_s2[ORDER.index(m)]
        marker = 's' if row_v > 0 else 'o'
        size = 200 if row_v > 0 else 180
        ax.scatter(row_v, row_c, c=C[m], s=size, marker=marker, edgecolors='#333',
                   linewidth=1.2 if m == 'aci_dqn' else 0.5, zorder=5, label=LABELS_SHORT[ORDER.index(m)])
    ax.set_xlabel('P3 SLA Violations', fontsize=10)
    ax.set_ylabel('Total Cost (CNY)', fontsize=10)
    ax.set_title('(h) S2 Cost vs. Violations', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5, loc='lower right', ncol=2)
    # Pareto frontier arrow
    ax.annotate('Optimal\n(lowest cost,\nzero violations)', xy=(0, min(costs_s2)),
                xytext=(max(viols_s2) * 0.6, min(costs_s2) * 1.06),
                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.8),
                fontsize=8, color='#9C27B0', fontweight='bold', ha='center')
    ax.grid(alpha=0.3)

    # Overall title
    fig.suptitle('S1 (Distribution Shift) & S2 (Burst Anomaly) — Method Comparison',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_stress_test.png", bbox_inches='tight')
    plt.close()
    print("  [OK] fig3_stress_test.png")


# ==============================================================================
# Figure 4: Conformal Prediction Diagnostics
# ==============================================================================
def fig4_conformal():
    """Conformal coverage + ACI alpha adaptation from S2 scenario."""
    from src.utils import load_config
    from src.data_preprocess import day_matrix, normalised_day_matrix
    from src.datacenter_env import DataCenterEnv, action_to_n
    from src.workload_generator import Task, DayWorkload
    from src.rl.dqn_agent import DQNAgent
    from src.rl.train_dqn import IdentityAugmenter
    from src.rl.augmenters import ConformalAugmenter

    cfg = load_config("config.yaml")
    K = cfg['qos']['K']
    T = cfg['time']['slots_per_day']

    # Load env
    processed = pd.read_csv("outputs/processed/processed_load.csv")
    raw, _ = day_matrix(processed, T)
    norm, _ = normalised_day_matrix(processed, T)

    # Recreate S2 scenario
    def s2_arrival_rates(t, k):
        if k == 0: return 3.0
        elif k == 1: return 4.0
        else: return 2.0

    rng = np.random.default_rng(2000)
    tasks_per_slot = [[[] for _ in range(T)] for _ in range(K)]
    service_mean = np.array(cfg['qos']['service_mean'])

    for t in range(T):
        for k in range(K):
            lam = s2_arrival_rates(t, k)
            if k == 2 and rng.random() < 0.20:
                lam *= rng.uniform(5, 15)
            n = rng.poisson(max(0.0, lam))
            if n == 0: continue
            for _ in range(n):
                dl = t + int(cfg['qos']['deadline_slots'][k])
                tasks_per_slot[k][t].append(Task(
                    arrival=t, deadline=min(T-1, dl), work=1.0, remaining=1.0))

    wl = DayWorkload(lam=np.zeros((K,T)), n_arrivals=np.zeros((K,T), dtype=int),
                     arrival_work=np.zeros((K,T)), tasks_per_slot=tasks_per_slot)

    # Run ACI and DtACI on this scenario, collecting per-slot intervals
    env_aci = DataCenterEnv(cfg=cfg, day_load_matrix=raw, day_norm_matrix=norm, base_seed=2024)
    aug_aci = ConformalAugmenter(cfg, learner='aci', use_shield=False)
    env_aci.reset(0)
    env_aci.workload = wl

    env_dt = DataCenterEnv(cfg=cfg, day_load_matrix=raw, day_norm_matrix=norm, base_seed=2024)
    aug_dt = ConformalAugmenter(cfg, learner='dtaci', use_shield=False)
    env_dt.reset(0)
    env_dt.workload = wl

    aci_data = []; dt_data = []
    for slot in range(T):
        # ACI
        s_aci = aug_aci.augment(env_aci.get_state(), env_aci)
        lo_a, hi_a = aug_aci.cp.intervals_h_steps()
        actual_a = sum(1 for _ in env_aci.workload.tasks_per_slot[2][slot]) if slot < T else 0
        env_aci.step(0)  # dummy step
        aci_data.append({
            'slot': slot, 'actual': actual_a,
            'lo': lo_a[2, 0], 'hi': hi_a[2, 0],
            'covered': lo_a[2, 0] <= actual_a <= hi_a[2, 0],
        })

        # DtACI
        s_dt = aug_dt.augment(env_dt.get_state(), env_dt)
        lo_d, hi_d = aug_dt.cp.intervals_h_steps()
        actual_d = sum(1 for _ in env_dt.workload.tasks_per_slot[2][slot]) if slot < T else 0
        env_dt.step(0)
        dt_data.append({
            'slot': slot, 'actual': actual_d,
            'lo': lo_d[2, 0], 'hi': hi_d[2, 0],
            'covered': lo_d[2, 0] <= actual_d <= hi_d[2, 0],
        })

    aci_df = pd.DataFrame(aci_data)
    dt_df = pd.DataFrame(dt_data)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for idx, (df, name, color, ax) in enumerate([
        (aci_df, 'ACI', '#9C27B0', axes[0]),
        (dt_df, 'DtACI', '#E91E63', axes[1]),
    ]):
        ax.fill_between(df['slot'], df['lo'], df['hi'], alpha=0.25, color=color, label=f'{name} interval')
        ax.plot(df['slot'], df['actual'], 'o-', color='#333', markersize=3, linewidth=1, label='Actual P3 arrivals')
        # Mark uncovered points
        uncovered = df[~df['covered']]
        if len(uncovered) > 0:
            ax.scatter(uncovered['slot'], uncovered['actual'], color='red', s=40, zorder=5,
                       marker='x', linewidth=1.5, label=f'Uncovered ({len(uncovered)})')
        cov = df['covered'].mean()
        ax.set_title(f'({chr(97+idx)}) {name}: P3 Conformal Intervals (coverage={cov:.1%})',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('P3 Arrivals per Slot', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, T - 1)

    axes[1].set_xlabel('Time Slot (15-min interval)', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_conformal_diagnostics.png")
    plt.close()
    print("  [OK] fig4_conformal_diagnostics.png")


# ==============================================================================
# Figure 5: Action Policy Analysis
# ==============================================================================
def fig5_action_policy():
    fig = plt.figure(figsize=(15, 10))

    # 5a: Server count distribution (box plot from daily results)
    ax1 = fig.add_subplot(2, 3, (1, 2))
    box_data = [daily[daily['method'] == m]['average_active_servers'].values for m in ORDER]
    bp = ax1.boxplot(box_data, patch_artist=True, widths=0.5)
    for patch, m in zip(bp['boxes'], ORDER):
        patch.set_facecolor(C[m])
        patch.set_alpha(0.75)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(1.5)
    ax1.set_xticklabels(LABELS_SHORT, fontsize=8)
    ax1.set_ylabel('Avg Active Servers per Day', fontsize=10)
    ax1.set_title('(a) Server Count Distribution (267 test days)', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 5b: Utilization distribution
    ax2 = fig.add_subplot(2, 3, 3)
    util_data = [daily[daily['method'] == m]['average_utilization'].values for m in ORDER]
    bp2 = ax2.boxplot(util_data, patch_artist=True, widths=0.5)
    for patch, m in zip(bp2['boxes'], ORDER):
        patch.set_facecolor(C[m])
        patch.set_alpha(0.75)
    for median in bp2['medians']:
        median.set_color('white')
        median.set_linewidth(1.5)
    ax2.set_xticklabels(LABELS_SHORT, fontsize=8, rotation=0)
    ax2.set_ylabel('Avg Utilization per Day', fontsize=10)
    ax2.set_title('(b) Utilization Distribution', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 5c: Action distribution from debug data
    ax3 = fig.add_subplot(2, 3, (4, 6))
    action_methods = ['dqn', 'aci_dqn', 'dtaci_dqn']
    action_labels = ['DQN', 'ACI-DQN', 'DtACI-DQN']
    action_colors = [C[m] for m in action_methods]
    width = 0.25
    x = np.arange(3)
    for i, (m, label, color) in enumerate(zip(action_methods, action_labels, action_colors)):
        try:
            ad = pd.read_csv(f'outputs/action_debug_{m}.csv')
            actions = ad['raw_action'].values
            mean_a = actions.mean()
            std_a = actions.std()
            ax3.bar(i, mean_a, width, color=color, alpha=0.8,
                    yerr=std_a, capsize=5, edgecolor='white', linewidth=0.5)
            ax3.text(i, mean_a + std_a + 0.5, f'{mean_a:.1f}\n±{std_a:.1f}',
                     ha='center', fontsize=8, fontweight='bold')
        except Exception:
            pass
    ax3.set_xticks(x)
    ax3.set_xticklabels(action_labels, fontsize=9)
    ax3.set_ylabel('Mean Raw Action Index (0-20)', fontsize=10)
    ax3.set_title('(c) RL Action Selection (3 sample days)', fontsize=11, fontweight='bold')
    ax3.axhline(y=10, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax3.text(2.5, 10.2, 'midpoint', fontsize=7, color='gray', ha='right')
    # Add server count mapping
    ax3_twin = ax3.twinx()
    Nmin, Nmax = 8, 120
    bins = 21
    bw = (Nmax - Nmin) / bins
    y_ticks = np.arange(0, 21, 5)
    y_labels = [f'{Nmin + int((y + 0.5) * bw)}' for y in y_ticks]
    ax3_twin.set_yticks(y_ticks)
    ax3_twin.set_yticklabels(y_labels, fontsize=7, color='#666')
    ax3_twin.set_ylabel('≈Server Count', fontsize=8, color='#666')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_action_policy.png")
    plt.close()
    print("  [OK] fig5_action_policy.png")


# ==============================================================================
if __name__ == "__main__":
    print("Generating publication-quality figures...")
    fig1_cost_efficiency()
    fig2_training_dynamics()
    fig3_stress_test()
    fig4_conformal()
    fig5_action_policy()
    print(f"\nAll figures saved to {FIG_DIR}/")
