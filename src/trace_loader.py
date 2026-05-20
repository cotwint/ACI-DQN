"""
src/trace_loader.py
-------------------
Load real Alibaba PAI cluster traces and convert them into
``DayWorkload`` objects consumable by ``DataCenterEnv``.

This module is the **real-task interface**. When ``workload.source`` in
config.yaml is set to ``"trace"``, the experiment pipeline calls
``load_trace_workloads()`` instead of using synthetic Poisson workloads.

Supported input format
----------------------
The Alibaba PAI trace consists of CSV files (no header):
  - pai_task_table.csv   : task metadata with start/end time, plan_cpu
  - pai_group_tag_table.csv : workload classification labels
  - pai_machine_spec.csv : machine capacities

Priority assignment heuristic
-----------------------------
  - GPU tasks / "delay_sensitive" workloads  → P1
  - Interactive / CPU-intensive tasks        → P2
  - Batch / "best_effort" workloads          → P3

To customise this mapping, override ``classify_priority()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .workload_generator import DayWorkload, Task


# ---------------------------------------------------------------------------
# Priority classification (override these for different trace formats)
# ---------------------------------------------------------------------------

def classify_priority(
    task_row: pd.Series,
    group_info: Optional[pd.DataFrame] = None,
) -> int:
    """Return 0-based priority index (0=P1, 1=P2, 2=P3) for a task row.

    Override this function to adapt to different trace schemas.
    """
    # Heuristic: GPU tasks → P1, else P2/P3 based on duration quantile
    gpu = task_row.get("plan_gpu", 0)
    gpu_type = str(task_row.get("gpu_type", ""))
    if pd.notna(gpu) and float(gpu) > 0:
        return 0  # GPU → P1
    if gpu_type and gpu_type not in ("nan", "", "None"):
        return 0

    # Check group info for workload label if available
    if group_info is not None and len(group_info) > 0:
        wl = str(group_info.iloc[0].get("workload", "")).lower()
        if any(kw in wl for kw in ("delay_sensitive", "online", "service")):
            return 0
        if any(kw in wl for kw in ("interactive",)):
            return 1
        if any(kw in wl for kw in ("batch", "best_effort", "offline")):
            return 2

    return 2  # default: P3


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_trace_workloads(
    trace_dir: str | Path,
    cfg: Dict,
    day_dates: List,
    time_slots: int = 96,
    start_timestamp: Optional[float] = None,
) -> Dict[int, DayWorkload]:
    """Load Alibaba PAI trace and return ``{day_index: DayWorkload}``.

    Parameters
    ----------
    trace_dir : path to directory containing the PAI CSV files.
    cfg : full config dict (for QoS deadline / service params).
    day_dates : list of ``datetime.date`` objects for the days to cover.
        Day 0 = earliest date in ``day_dates``.
    time_slots : slots per day (default 96 for 15-min).
    start_timestamp : optional epoch seconds anchoring the trace timeline.
        If None, the trace's earliest start_time is used.

    Returns
    -------
    workloads : dict mapping day_index -> DayWorkload.
        Keys are 0-based indices into ``day_dates``.
    """
    trace_dir = Path(trace_dir)
    task_path = trace_dir / "pai_task_table.csv"
    group_path = trace_dir / "pai_group_tag_table.csv"

    if not task_path.exists():
        raise FileNotFoundError(
            f"Task table not found at {task_path}. "
            f"Place PAI trace CSVs in {trace_dir}/"
        )

    # Column names (no header in PAI trace)
    task_cols = [
        "job_name", "task_name", "inst_num", "status",
        "start_time", "end_time", "plan_cpu", "plan_mem",
        "plan_gpu", "gpu_type",
    ]
    group_cols = ["inst_id", "user", "gpu_type_spec", "group", "workload"]

    # Read task table
    tasks_df = pd.read_csv(task_path, names=task_cols, low_memory=False)
    tasks_df = tasks_df.dropna(subset=["start_time"])

    # Anchor time to the trace's earliest start_time
    if start_timestamp is None:
        start_timestamp = float(tasks_df["start_time"].min())

    # Offset tasks to align with our day grid
    tasks_df["t_offset"] = tasks_df["start_time"] - start_timestamp
    tasks_df["slot_abs"] = (tasks_df["t_offset"] / 900).astype(int)  # 900s = 15min
    tasks_df["day_index"] = tasks_df["slot_abs"] // time_slots
    tasks_df["slot_in_day"] = tasks_df["slot_abs"] % time_slots

    # Load group tags for priority classification
    group_lookup: Dict[int, pd.DataFrame] = {}
    if group_path.exists():
        groups_df = pd.read_csv(group_path, names=group_cols, low_memory=False)
        for _, g_row in groups_df.iterrows():
            iid = int(g_row["inst_id"]) if pd.notna(g_row["inst_id"]) else -1
            if iid >= 0:
                group_lookup[iid] = g_row

    K = int(cfg["qos"]["K"])
    deadlines = np.asarray(cfg["qos"]["deadline_slots"], dtype=int)
    mu = np.asarray(cfg["qos"]["service_mean"], dtype=np.float64)

    num_days = len(day_dates)
    max_day_idx = num_days - 1

    # Filter to relevant day range
    tasks_df = tasks_df[
        (tasks_df["day_index"] >= 0) & (tasks_df["day_index"] <= max_day_idx)
    ]

    # Pre-allocate empty workloads for all days
    workloads: Dict[int, DayWorkload] = {}
    for d in range(num_days):
        tasks_per_slot: List[List[List[Task]]] = [
            [[] for _ in range(time_slots)] for _ in range(K)
        ]
        arrival_work = np.zeros((K, time_slots), dtype=np.float64)
        n_arrivals = np.zeros((K, time_slots), dtype=int)
        lam = np.zeros((K, time_slots), dtype=np.float64)
        workloads[d] = DayWorkload(
            lam=lam,
            n_arrivals=n_arrivals,
            arrival_work=arrival_work,
            tasks_per_slot=tasks_per_slot,
        )

    # Process tasks and insert into workloads
    processed = 0
    for _, row in tasks_df.iterrows():
        day_idx = int(row["day_index"])
        slot = int(row["slot_in_day"])
        if day_idx < 0 or day_idx > max_day_idx:
            continue
        if slot < 0 or slot >= time_slots:
            continue

        # Priority
        inst_num = int(row["inst_num"]) if pd.notna(row.get("inst_num")) else -1
        gi = group_lookup.get(inst_num)
        k = classify_priority(row, gi)
        k = int(np.clip(k, 0, K - 1))

        # Compute requirement (from plan_cpu, normalised)
        cpu = row.get("plan_cpu", 0)
        if pd.isna(cpu) or float(cpu) <= 0:
            cpu = float(mu[k])
        else:
            cpu = float(cpu) * 0.01  # scale plan_cpu to compute units

        # Duration → deadline
        end = row.get("end_time", 0)
        if pd.notna(end) and float(end) > 0:
            dur_s = float(end) - float(row["start_time"])
            dur_slots = max(1, int(np.ceil(dur_s / 900)))
        else:
            dur_slots = deadlines[k]
        dl = min(time_slots - 1, slot + max(1, dur_slots))

        wl = workloads[day_idx]
        task = Task(arrival=slot, deadline=dl, work=cpu, remaining=cpu)
        wl.tasks_per_slot[k][slot].append(task)
        wl.arrival_work[k, slot] += cpu
        wl.n_arrivals[k, slot] += 1
        wl.lam[k, slot] += 1.0
        processed += 1

    print(f"TraceLoader: loaded {processed} tasks across {num_days} days")
    return workloads
