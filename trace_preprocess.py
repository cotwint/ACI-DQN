"""
trace_preprocess.py
==================
Preprocess Alibaba PAI GPU cluster trace v2020 into the DayWorkload
format used by the datacenter scheduler environment.

Input:  trace_dataset/pai_*.csv (no headers, column names from .header files)
Output: outputs/trace/processed_trace_days.pkl  (list of DayWorkload objects)
        outputs/trace/trace_summary.csv          (per-day statistics)

Priority Classification (P1/P2/P3):
  Uses workload type (from group_tag) as primary signal,
  duration and GPU count as secondary signals.

  P1 (紧急): interactive / time-sensitive workloads (ctr, short inference)
  P2 (交互): moderate training tasks (nmt, graphlearn, mid-duration)
  P3 (批处理): long training jobs (bert, xlnet, resnet, inception, vgg, rl)

Deadline mapping:
  P1 → 2 slots (30 min)  - penalty 50 CNY
  P2 → 8 slots (2 hours) - penalty 20 CNY
  P3 → 32 slots (8 hours)- penalty 5 CNY

Work units:
  work = (plan_gpu + 0.1 * plan_cpu/100) * duration_slots
  Clamped to reasonable range per priority.
"""

import sys
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column names from .header files
# ---------------------------------------------------------------------------
TASK_COLS = ['job_name', 'task_name', 'inst_num', 'status', 'start_time',
             'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu', 'gpu_type']
JOB_COLS = ['job_name', 'inst_id', 'user', 'status', 'start_time', 'end_time']
GROUP_COLS = ['inst_id', 'user', 'gpu_type_spec', 'group', 'workload']
INST_COLS = ['job_name', 'task_name', 'inst_name', 'worker_name', 'inst_id',
             'status', 'start_time', 'end_time', 'machine']
MACHINE_COLS = ['machine', 'gpu_type', 'cap_cpu', 'cap_mem', 'cap_gpu']

# ---------------------------------------------------------------------------
# Workload → priority mapping
# ---------------------------------------------------------------------------
WORKLOAD_PRIORITY = {
    # P1: time-sensitive, interactive
    'ctr': 1,
    # P2: moderate, mixed
    'nmt': 2,
    'graphlearn': 2,
    'rl': 2,
    # P3: heavy training (long running)
    'bert': 3,
    'xlnet': 3,
    'resnet': 3,
    'inception': 3,
    'vgg': 3,
}

# ---------------------------------------------------------------------------
# Priority parameters (matching original config)
# ---------------------------------------------------------------------------
DEADLINE_SLOTS = {1: 2, 2: 8, 3: 32}        # slots (15-min each)
SLA_PENALTY = {1: 50.0, 2: 20.0, 3: 5.0}   # CNY per violation
SLOT_SECONDS = 900                            # 15 minutes in seconds

# ---------------------------------------------------------------------------
# Data structures (compatible with workload_generator.py)
# ---------------------------------------------------------------------------
@dataclass
class Task:
    arrival: int
    deadline: int
    work: float
    remaining: float

@dataclass
class DayWorkload:
    lam: np.ndarray
    n_arrivals: np.ndarray
    arrival_work: np.ndarray
    tasks_per_slot: list


def load_trace_tables(trace_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all trace tables, joining to get full task info."""
    print("Loading trace tables...")

    # Read task table (largest, ~1.26M rows)
    tasks = pd.read_csv(
        Path(trace_dir) / 'pai_task_table.csv',
        names=TASK_COLS,
        dtype={'job_name': str, 'task_name': str, 'status': str,
               'gpu_type': str, 'plan_cpu': float, 'plan_mem': float,
               'plan_gpu': float, 'inst_num': float}
    )
    print(f"  task_table: {len(tasks):,} rows")

    # Read job table to link job_name → inst_id
    jobs = pd.read_csv(
        Path(trace_dir) / 'pai_job_table.csv',
        names=JOB_COLS,
        dtype={'job_name': str, 'inst_id': str, 'user': str, 'status': str}
    )
    print(f"  job_table:  {len(jobs):,} rows")

    # Read group tag for workload type
    groups = pd.read_csv(
        Path(trace_dir) / 'pai_group_tag_table.csv',
        names=GROUP_COLS,
        dtype={'inst_id': str, 'user': str, 'gpu_type_spec': str,
               'group': str, 'workload': str}
    )
    print(f"  group_tag:  {len(groups):,} rows")

    return {'tasks': tasks, 'jobs': jobs, 'groups': groups}


def classify_priority(workload: str, duration_sec: float,
                      plan_gpu: float, plan_cpu: float,
                      dur_p33: float, dur_p66: float) -> int:
    """Classify task into P1/P2/P3.

    Primary: workload type label (covers ~15% of tasks).
    Fallback: duration percentile thresholds computed globally.
    """
    wl = str(workload).lower().strip()

    if wl in WORKLOAD_PRIORITY:
        return WORKLOAD_PRIORITY[wl]

    # Fallback: duration-based split
    if duration_sec < dur_p33:
        return 1
    elif duration_sec < dur_p66:
        return 2
    else:
        return 3


def compute_work(plan_gpu: float, plan_cpu: float,
                 duration_sec: float, deadline_slots: int) -> float:
    """Compute work as server-slots needed.

    Work = effective_resource * min(task_duration, deadline) in slot units.
    Scaled so that ~70 servers at 75% util can handle a typical day.
    """
    gpu = min(plan_gpu if pd.notna(plan_gpu) and plan_gpu > 0 else 0.0, 8.0)
    cpu = plan_cpu if pd.notna(plan_cpu) and plan_cpu > 0 else 0.0

    # Cap duration to deadline (tasks can't use more than their SLA window)
    dur = min(duration_sec / 900.0, deadline_slots)  # in slot units

    if gpu > 0:
        work = gpu * dur * 0.15
    else:
        work = (cpu / 600.0) * dur * 0.15

    return max(0.05, min(work, 50.0))


def assign_deadline_slot(arrival_slot: int, duration_sec: float,
                         priority: int, slots_per_day: int) -> int:
    """Assign deadline slot index.

    Uses the nominal deadline for the priority as the SLA deadline.
    Tasks that actually run longer than the nominal deadline may still
    be scheduled but at risk of violation.
    """
    nominal_slots = DEADLINE_SLOTS[priority]
    deadline = arrival_slot + nominal_slots
    return min(slots_per_day - 1, int(deadline))


def process_trace(trace_dir: str,
                  output_dir: str,
                  slots_per_day: int = 96,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[DayWorkload]:
    """Main preprocessing pipeline.

    Parameters
    ----------
    trace_dir : str
        Path to directory containing pai_*.csv files.
    output_dir : str
        Path for processed output.
    slots_per_day : int
        Number of 15-min slots per day (default 96).
    start_time, end_time : float or None
        Subset the trace to this time range (seconds). None = use all.

    Returns
    -------
    List of DayWorkload, one per complete day.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = load_trace_tables(trace_dir)
    tasks = tables['tasks']
    jobs = tables['jobs']
    groups = tables['groups']

    # --- Step 1: Filter Terminated tasks ---
    print("\nFiltering Terminated tasks...")
    tasks = tasks[tasks['status'] == 'Terminated'].copy()
    tasks = tasks.dropna(subset=['start_time', 'end_time'])
    # Remove zero/negative durations
    tasks = tasks[tasks['end_time'] > tasks['start_time']]
    print(f"  {len(tasks):,} Terminated tasks with valid duration")

    # --- Step 2: Join with jobs → inst_id ---
    print("\nJoining task → job → group_tag...")
    # Keep one row per (job_name) for joining (deduplicate jobs on job_name)
    jobs_dedup = jobs.drop_duplicates(subset='job_name', keep='first')
    tasks = tasks.merge(jobs_dedup[['job_name', 'inst_id']],
                        on='job_name', how='left')

    # Join with groups → workload type
    groups_dedup = groups.drop_duplicates(subset='inst_id', keep='first')
    tasks = tasks.merge(groups_dedup[['inst_id', 'workload', 'group']],
                        on='inst_id', how='left')
    tasks['workload'] = tasks['workload'].fillna('unknown')
    print(f"  {len(tasks):,} tasks after join")
    print(f"  workload types: {tasks['workload'].value_counts().head(15).to_dict()}")

    # --- Step 3: Time range ---
    t_min = tasks['start_time'].min() if start_time is None else start_time
    t_max = tasks['end_time'].max() if end_time is None else end_time
    if start_time is not None:
        tasks = tasks[tasks['start_time'] >= start_time]
    if end_time is not None:
        tasks = tasks[tasks['start_time'] < end_time]
    print(f"\nTime range: {t_min:.0f} → {t_max:.0f} ({t_max - t_min:.0f} seconds)")

    # --- Step 4: Compute task-level features ---
    print("\nComputing task features...")
    tasks['duration_sec'] = tasks['end_time'] - tasks['start_time']
    tasks['arrival_slot'] = ((tasks['start_time'] - t_min) / SLOT_SECONDS).astype(int)

    # Duration percentiles for fallback priority classification
    dur_p33 = tasks['duration_sec'].quantile(0.33)
    dur_p66 = tasks['duration_sec'].quantile(0.66)
    print(f"  Duration percentiles: P33={dur_p33:.0f}s, P66={dur_p66:.0f}s")

    # Classify priority (needs duration percentiles for fallback)
    tasks['priority'] = tasks.apply(
        lambda r: classify_priority(r['workload'], r['duration_sec'],
                                    r['plan_gpu'], r['plan_cpu'],
                                    dur_p33, dur_p66), axis=1)

    # Compute work and deadline
    tasks['deadline_nominal'] = tasks['priority'].map(DEADLINE_SLOTS)
    total_slots = int((t_max - t_min) / SLOT_SECONDS)
    tasks['deadline_slot'] = tasks.apply(
        lambda r: assign_deadline_slot(r['arrival_slot'], r['duration_sec'],
                                       r['priority'], total_slots), axis=1)

    tasks['work'] = tasks.apply(
        lambda r: compute_work(r['plan_gpu'], r['plan_cpu'],
                               r['duration_sec'], r['deadline_nominal']), axis=1)

    # --- Step 5: Statistics ---
    print(f"\nPriority distribution:")
    for p in [1, 2, 3]:
        pt = tasks[tasks['priority'] == p]
        print(f"  P{p}: {len(pt):,} tasks, "
              f"work mean={pt['work'].mean():.2f}, "
              f"duration median={pt['duration_sec'].median():.0f}s, "
              f"GPU mean={pt['plan_gpu'].mean():.1f}, "
              f"CPU mean={pt['plan_cpu'].mean():.0f}")

    # --- Step 6: Aggregate into slots ---
    print(f"\nAggregating into {total_slots} slots...")
    K = 3
    # Initialize per-slot task lists
    tasks_per_slot_global = [[[] for _ in range(total_slots)] for _ in range(K)]
    lam = np.zeros((K, total_slots), dtype=np.float64)
    n_arrivals = np.zeros((K, total_slots), dtype=np.int32)
    arrival_work = np.zeros((K, total_slots), dtype=np.float64)

    for _, row in tasks.iterrows():
        k = int(row['priority']) - 1  # 0-indexed
        s = int(row['arrival_slot'])
        if s >= total_slots:
            continue
        dl = int(row['deadline_slot'])
        w = float(row['work'])

        task = Task(arrival=s, deadline=min(dl, total_slots - 1),
                    work=w, remaining=w)
        tasks_per_slot_global[k][s].append(task)
        lam[k, s] += 1.0
        n_arrivals[k, s] += 1
        arrival_work[k, s] += w

    # --- Step 7: Split into days ---
    print("\nSplitting into days...")
    n_days = total_slots // slots_per_day
    print(f"  {n_days} complete days ({total_slots} slots / {slots_per_day})")

    daily_summary = []
    workloads = []

    for d in range(n_days):
        t0 = d * slots_per_day
        t1 = t0 + slots_per_day

        day_lam = lam[:, t0:t1].copy()
        day_n = n_arrivals[:, t0:t1].copy()
        day_work = arrival_work[:, t0:t1].copy()

        # Offset task arrivals and deadlines to day-relative
        day_tasks = [[[] for _ in range(slots_per_day)] for _ in range(K)]
        for k in range(K):
            for s in range(slots_per_day):
                for task in tasks_per_slot_global[k][t0 + s]:
                    day_tasks[k][s].append(Task(
                        arrival=s,
                        deadline=max(s, min(slots_per_day - 1,
                                           task.deadline - t0)),
                        work=task.work,
                        remaining=task.work
                    ))

        wl = DayWorkload(
            lam=day_lam,
            n_arrivals=day_n,
            arrival_work=day_work,
            tasks_per_slot=day_tasks,
        )
        workloads.append(wl)

        # Summary
        daily_summary.append({
            'day': d,
            'P1_tasks': int(day_n[0].sum()),
            'P2_tasks': int(day_n[1].sum()),
            'P3_tasks': int(day_n[2].sum()),
            'total_tasks': int(day_n.sum()),
            'P1_work': float(day_work[0].sum()),
            'P2_work': float(day_work[1].sum()),
            'P3_work': float(day_work[2].sum()),
            'total_work': float(day_work.sum()),
            'P1_mean_work': float(day_work[0, day_n[0] > 0].mean()) if day_n[0].sum() > 0 else 0,
            'P2_mean_work': float(day_work[1, day_n[1] > 0].mean()) if day_n[1].sum() > 0 else 0,
            'P3_mean_work': float(day_work[2, day_n[2] > 0].mean()) if day_n[2].sum() > 0 else 0,
        })

    # --- Step 8: Save ---
    print("\nSaving...")
    # Save workloads
    with open(out / 'processed_trace_days.pkl', 'wb') as f:
        pickle.dump(workloads, f)
    print(f"  Saved {len(workloads)} days to {out / 'processed_trace_days.pkl'}")

    # Save summary
    summary_df = pd.DataFrame(daily_summary)
    summary_df.to_csv(out / 'trace_summary.csv', index=False)
    print(f"  Saved summary to {out / 'trace_summary.csv'}")

    # Save priority stats
    overall = summary_df.describe()
    print(f"\n=== Daily Summary Statistics ===")
    print(f"  Tasks/day: {summary_df['total_tasks'].mean():.0f} ± {summary_df['total_tasks'].std():.0f}")
    print(f"  P1/day:    {summary_df['P1_tasks'].mean():.0f} ± {summary_df['P1_tasks'].std():.0f}")
    print(f"  P2/day:    {summary_df['P2_tasks'].mean():.0f} ± {summary_df['P2_tasks'].std():.0f}")
    print(f"  P3/day:    {summary_df['P3_tasks'].mean():.0f} ± {summary_df['P3_tasks'].std():.0f}")
    print(f"  Work/day:  {summary_df['total_work'].mean():.0f} ± {summary_df['total_work'].std():.0f}")

    return workloads


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', default='trace_dataset')
    parser.add_argument('--output-dir', default='outputs/trace')
    parser.add_argument('--slots-per-day', type=int, default=96)
    parser.add_argument('--start-time', type=float, default=None)
    parser.add_argument('--end-time', type=float, default=None)
    args = parser.parse_args()

    wls = process_trace(
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        slots_per_day=args.slots_per_day,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(f"\nDone. {len(wls)} days ready for evaluation.")
