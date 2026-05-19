"""
src/data_preprocess.py
----------------------
Read the regional load CSV (Chinese filename / Chinese columns), clean it
to a 15-minute grid, keep only complete days (96 slots each) and split
chronologically into train / calibration / test.

IMPORTANT: this is *regional power load* data, **not** an Alibaba cluster
trace. We will use its normalised shape later to drive synthetic task
arrival rates -- see ``workload_generator.py``.

Outputs (written to ``cfg['paths']['processed_dir']``):

* ``processed_load.csv``        -- cleaned 15-min series with 'date'
* ``complete_days_summary.csv`` -- one row per kept day
* ``train_cal_test_split.csv``  -- per-day split label

No interaction with test-set data is performed beyond labelling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger, load_config


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV with Chinese column names, return tidy DataFrame.

    Returns
    -------
    DataFrame with columns: ``timestamp`` (datetime64[ns]) and
    ``regional_load_kw`` (float).
    """
    # The file uses utf-8 with bom variants -- be permissive.
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Cannot decode {csv_path} with utf-8 or gbk")

    # Identify the timestamp + load columns by Chinese names with a
    # tolerant fallback by position.
    time_col = next((c for c in df.columns if "时间" in c), df.columns[1])
    load_col = next((c for c in df.columns if "有功" in c or "kw" in c.lower()),
                    df.columns[-1])

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[time_col], errors="coerce"),
        "regional_load_kw": pd.to_numeric(df[load_col], errors="coerce"),
    })
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Cleaning and aligning to 15-minute grid
# ---------------------------------------------------------------------------

def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamps, sort, reindex onto a 15-min grid and
    linearly interpolate small gaps.
    """
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    full_idx = pd.date_range(df.index.min().floor("15min"),
                             df.index.max().ceil("15min"),
                             freq="15min")
    df = df.reindex(full_idx)
    df.index.name = "timestamp"
    df["regional_load_kw"] = df["regional_load_kw"].interpolate(
        method="linear", limit=4, limit_direction="both"
    )
    df = df.reset_index()
    return df


def keep_complete_days(df: pd.DataFrame,
                       slots_per_day: int = 96) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only those calendar days that have all ``slots_per_day`` slots
    with non-null values. Return (per-slot df, per-day summary)."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    grp = df.groupby("date")
    summary = grp["regional_load_kw"].agg(
        n_slots="count",
        load_min="min",
        load_max="max",
        load_mean="mean",
    ).reset_index()
    summary["complete"] = summary["n_slots"] == slots_per_day

    good_days = set(summary.loc[summary["complete"], "date"])
    kept = df[df["date"].isin(good_days)].copy()

    # Add slot index 0..95 for convenience.
    kept["slot"] = (kept["timestamp"].dt.hour * 4
                    + kept["timestamp"].dt.minute // 15)
    kept = kept.sort_values(["date", "slot"]).reset_index(drop=True)
    return kept, summary


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(days: pd.Series,
                        train_ratio: float,
                        cal_ratio: float) -> pd.DataFrame:
    """Assign 'train' / 'calibration' / 'test' to each unique day in
    chronological order.  Test ratio = 1 - train - cal.
    """
    unique_days = sorted(days.unique())
    n = len(unique_days)
    n_train = int(n * train_ratio)
    n_cal = int(n * cal_ratio)
    n_test = n - n_train - n_cal
    labels = (["train"] * n_train
              + ["calibration"] * n_cal
              + ["test"] * n_test)
    return pd.DataFrame({"date": unique_days, "split": labels})


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run(cfg: Dict) -> Dict[str, str]:
    log = get_logger("preprocess")
    raw_csv = cfg["paths"]["raw_csv"]
    out_dir = ensure_dir(cfg["paths"]["processed_dir"])

    log.info(f"Reading raw load CSV: {raw_csv}")
    raw = load_raw_csv(raw_csv)
    log.info(f"  raw rows = {len(raw)}, "
             f"range = {raw['timestamp'].min()} .. {raw['timestamp'].max()}")

    log.info("Cleaning and resampling to 15-min grid ...")
    clean = clean_and_resample(raw)

    log.info("Filtering to complete days ...")
    kept, summary = keep_complete_days(clean, cfg["time"]["slots_per_day"])
    log.info(f"  kept {summary['complete'].sum()} / {len(summary)} days")

    # Split by day.
    split = chronological_split(
        kept["date"],
        cfg["split"]["train_ratio"],
        cfg["split"]["cal_ratio"],
    )

    processed_path = out_dir / "processed_load.csv"
    summary_path = out_dir / "complete_days_summary.csv"
    split_path = out_dir / "train_cal_test_split.csv"

    kept.to_csv(processed_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    split.to_csv(split_path, index=False, encoding="utf-8")

    log.info(f"  wrote {processed_path}")
    log.info(f"  wrote {summary_path}")
    log.info(f"  wrote {split_path}")
    return {
        "processed_load": str(processed_path),
        "complete_days_summary": str(summary_path),
        "split": str(split_path),
    }


# ---------------------------------------------------------------------------
# Convenience helpers for downstream code
# ---------------------------------------------------------------------------

def load_processed(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reload the processed load and the per-day split labels."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    load = pd.read_csv(proc_dir / "processed_load.csv",
                       parse_dates=["timestamp"])
    split = pd.read_csv(proc_dir / "train_cal_test_split.csv",
                        parse_dates=["date"])
    load["date"] = pd.to_datetime(load["date"]).dt.date
    split["date"] = pd.to_datetime(split["date"]).dt.date
    return load, split


def day_matrix(load: pd.DataFrame,
               slots_per_day: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """Return (D, slots_per_day) load matrix and an array of dates.

    Each row is one calendar day's regional load curve (kW).
    """
    dates = sorted(load["date"].unique())
    arr = np.zeros((len(dates), slots_per_day), dtype=np.float64)
    for i, d in enumerate(dates):
        chunk = load[load["date"] == d].sort_values("slot")
        if len(chunk) != slots_per_day:
            raise ValueError(f"Day {d} has {len(chunk)} slots, expected {slots_per_day}")
        arr[i] = chunk["regional_load_kw"].to_numpy()
    return arr, np.array(dates)


def normalised_day_matrix(load: pd.DataFrame,
                          slots_per_day: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """Same as ``day_matrix`` but every row scaled to [0,1] via min-max."""
    raw, dates = day_matrix(load, slots_per_day)
    lo = raw.min(axis=1, keepdims=True)
    hi = raw.max(axis=1, keepdims=True)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    norm = (raw - lo) / span
    return norm, dates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)
