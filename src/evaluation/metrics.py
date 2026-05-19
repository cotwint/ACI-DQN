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
