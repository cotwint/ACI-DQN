"""Aggregate per-episode stats into summary DataFrames."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..rl.train_dqn import EpisodeStats


PRIORITY_NAMES = ["P1", "P2", "P3"]


def episode_stats_to_row(method: str, stats: EpisodeStats) -> Dict:
    completed = np.where(stats.completed > 0, stats.completed, 1)
    sla_rate = stats.sla_violations / completed
    sla_count = stats.sla_violations
    avg_delay = stats.delay_sum / completed
    # Per-priority SLA cost from accumulated step tracking.
    pp_cost = (stats.overdue_pending
               if stats.overdue_pending is not None
               else np.zeros(3))
    return {
        "method": method,
        "day_index": stats.day_index,
        "total_objective_cost": stats.total_cost,
        "electricity_cost": stats.elec_cost,
        "qos_cost": stats.qos_cost,
        "switching_cost": stats.switch_cost,
        "total_energy_kwh": stats.energy,
        "peak_power_kw": stats.peak_power,
        "average_power_kw": stats.avg_power,
        "average_active_servers": stats.avg_n_active,
        "average_utilization": stats.avg_util,
        "P1_sla_violation_rate": float(sla_rate[0]),
        "P2_sla_violation_rate": float(sla_rate[1]),
        "P3_sla_violation_rate": float(sla_rate[2]),
        "P1_violation_count": float(sla_count[0]),
        "P2_violation_count": float(sla_count[1]),
        "P3_violation_count": float(sla_count[2]),
        "P1_sla_cost": float(pp_cost[0]),
        "P2_sla_cost": float(pp_cost[1]),
        "P3_sla_cost": float(pp_cost[2]),
        "P1_avg_delay": float(avg_delay[0]),
        "P2_avg_delay": float(avg_delay[1]),
        "P3_avg_delay": float(avg_delay[2]),
        "P1_completed": float(stats.completed[0]),
        "P2_completed": float(stats.completed[1]),
        "P3_completed": float(stats.completed[2]),
    }


def aggregate_to_summary(daily: pd.DataFrame,
                         extra_metrics: Dict[str, Dict] | None = None
                         ) -> pd.DataFrame:
    """Compute mean across evaluation days grouped by method."""
    # Count eval days per method.
    day_counts = daily.groupby("method")["day_index"].nunique().rename("num_eval_days")

    agg_cols = [c for c in daily.columns
                if c not in ("method", "day_index")]
    summary = daily.groupby("method")[agg_cols].mean().reset_index()

    # Merge day count.
    summary = summary.merge(day_counts, on="method", how="left")

    summary = summary.sort_values("total_objective_cost").reset_index(drop=True)

    if extra_metrics:
        rows = []
        for method, m in extra_metrics.items():
            row = {"method": method}
            cp = m.get("conformal", {})
            covs, widths = [], []
            for k_name, kd in cp.items():
                if kd.get("empirical_coverage") is not None:
                    covs.append(kd["empirical_coverage"])
                    widths.append(kd["avg_width"])
            row["conformal_coverage"] = float(np.mean(covs)) if covs else float("nan")
            row["conformal_avg_width"] = float(np.mean(widths)) if widths else float("nan")
            row["shield_mod_rate"] = m.get("shield_mod_rate", float("nan"))
            row["shield_avg_mod_size"] = m.get("shield_avg_mod_size", float("nan"))
            rows.append(row)
        extra_df = pd.DataFrame(rows)
        summary = summary.merge(extra_df, on="method", how="left")
    return summary


# ---------------------------------------------------------------------------
# Multi-seed aggregation with confidence intervals
# ---------------------------------------------------------------------------

def aggregate_with_ci(daily: pd.DataFrame,
                      group_cols: List[str] | None = None,
                      value_col: str = "total_objective_cost",
                      ) -> pd.DataFrame:
    """Compute mean, std, and 95 % CI across seeds for each method.

    Parameters
    ----------
    daily : DataFrame with one row per method/day/seed.
    group_cols : columns to group by before aggregating (default: ["method"]).
    value_col : primary cost column.

    Returns
    -------
    summary : DataFrame with columns like ``mean_cost``, ``std_cost``,
        ``ci_lower``, ``ci_upper``, ``mean_utilization``, etc.
    """
    if group_cols is None:
        group_cols = ["method"]

    # First average over days within each (method, scenario_seed, training_seed)
    day_agg_keys = group_cols + ["scenario_seed", "training_seed"]
    day_agg_keys = [c for c in day_agg_keys if c in daily.columns]

    metric_cols = [
        "total_objective_cost", "electricity_cost", "qos_cost",
        "switching_cost", "average_utilization", "average_active_servers",
        "P1_sla_violation_rate", "P2_sla_violation_rate",
        "P3_sla_violation_rate", "total_energy_kwh", "peak_power_kw",
    ]
    metric_cols = [c for c in metric_cols if c in daily.columns]

    per_seed = daily.groupby(day_agg_keys)[metric_cols].mean().reset_index()

    # Now aggregate over seeds
    rows = []
    for method in per_seed["method"].unique():
        mdata = per_seed[per_seed["method"] == method]
        row = {"method": method, "n_seeds": len(mdata)}
        for mc in metric_cols:
            vals = mdata[mc].dropna().values
            if len(vals) == 0:
                row[f"mean_{mc}"] = float("nan")
                row[f"std_{mc}"] = float("nan")
                row[f"ci_lower_{mc}"] = float("nan")
                row[f"ci_upper_{mc}"] = float("nan")
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            # 95% CI via t-distribution
            from scipy import stats as sp_stats
            if len(vals) > 1:
                se = std / np.sqrt(len(vals))
                ci = sp_stats.t.interval(0.95, df=len(vals) - 1, loc=mean, scale=se)
            else:
                ci = (mean, mean)
            row[f"mean_{mc}"] = mean
            row[f"std_{mc}"] = std
            row[f"ci_lower_{mc}"] = float(ci[0])
            row[f"ci_upper_{mc}"] = float(ci[1])
        rows.append(row)

    summary = pd.DataFrame(rows)
    # Sort by mean cost ascending
    cost_col = "mean_total_objective_cost"
    if cost_col in summary.columns:
        summary = summary.sort_values(cost_col).reset_index(drop=True)
    return summary


def compute_paired_tests(daily: pd.DataFrame,
                         group_col: str = "method",
                         value_col: str = "total_objective_cost",
                         ) -> pd.DataFrame:
    """Compute paired-difference tests between selected method pairs.

    Pairs: each method vs DQN, vs Forecast-DQN, vs Static-Conformal-DQN,
    vs Conformal-Greedy.

    Uses per-seed means as paired observations.
    """
    from scipy import stats as sp_stats

    # Per-seed means
    day_agg_keys = ["method", "scenario_seed", "training_seed"]
    day_agg_keys = [c for c in day_agg_keys if c in daily.columns]
    metric_cols = [value_col, "P1_sla_violation_rate",
                   "P3_sla_violation_rate", "average_utilization"]
    metric_cols = [c for c in metric_cols if c in daily.columns]
    per_seed = daily.groupby(day_agg_keys)[metric_cols].mean().reset_index()

    reference_methods = ["dqn", "forecast_dqn",
                         "static_conformal_dqn", "conformal_greedy"]
    all_methods = per_seed["method"].unique()

    rows = []
    for method_a in all_methods:
        for method_b in reference_methods:
            if method_a == method_b or method_b not in all_methods:
                continue
            m_a = per_seed[per_seed["method"] == method_a]
            m_b = per_seed[per_seed["method"] == method_b]
            # Merge on seeds for paired comparison
            seed_keys = [c for c in day_agg_keys if c != "method"]
            merged = m_a.merge(m_b, on=seed_keys, suffixes=("_a", "_b"))
            if len(merged) < 2:
                continue
            for mc in metric_cols:
                diffs = merged[f"{mc}_a"].values - merged[f"{mc}_b"].values
                mean_diff = float(np.mean(diffs))
                std_diff = float(np.std(diffs, ddof=1))
                se = std_diff / np.sqrt(len(diffs))
                ci = sp_stats.t.interval(0.95, df=len(diffs) - 1,
                                         loc=mean_diff, scale=se)
                t_stat, p_val = sp_stats.ttest_rel(
                    merged[f"{mc}_a"].values, merged[f"{mc}_b"].values,
                )
                rows.append({
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": mc,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "ci_lower": float(ci[0]),
                    "ci_upper": float(ci[1]),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                    "n_pairs": len(diffs),
                })

    return pd.DataFrame(rows)
