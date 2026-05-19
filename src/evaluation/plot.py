"""Matplotlib helpers for generating report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar_compare(summary: pd.DataFrame,
                column: str,
                title: str,
                ylabel: str,
                path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    df = summary[["method", column]].copy()
    df = df.sort_values(column)
    ax.bar(df["method"], df[column], edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grouped_sla_bar(summary: pd.DataFrame,
                    path: str | Path) -> None:
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(x - width, summary["P1_sla_violation_rate"], width, label="P1")
    ax.bar(x,         summary["P2_sla_violation_rate"], width, label="P2")
    ax.bar(x + width, summary["P3_sla_violation_rate"], width, label="P3")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_title("SLA Violation Rate by Priority")
    ax.set_ylabel("Violation rate")
    ax.set_xlabel("Method")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def power_curve_comparison(per_method_curves: Dict[str, np.ndarray],
                           path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for name, arr in per_method_curves.items():
        ax.plot(arr, label=name)
    ax.set_title("Facility Power Curve")
    ax.set_xlabel("15-min slot")
    ax.set_ylabel("Facility power (kW)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def active_server_comparison(per_method_curves: Dict[str, np.ndarray],
                             path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for name, arr in per_method_curves.items():
        ax.plot(arr, label=name)
    ax.set_title("Active Server Count")
    ax.set_xlabel("15-min slot")
    ax.set_ylabel("Active servers")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def queue_length_curve(per_method_queues: Dict[str, np.ndarray],
                       priority: int,
                       path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for name, arr in per_method_queues.items():
        ax.plot(arr, label=name)
    ax.set_title(f"P{priority} Queue Length")
    ax.set_xlabel("15-min slot")
    ax.set_ylabel("Queue length (tasks)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dtaci_coverage_width(coverage_log: List[int],
                         width_log: List[float],
                         target_alpha: float,
                         path: str | Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    if coverage_log:
        running = np.cumsum(coverage_log) / np.arange(1, len(coverage_log) + 1)
        ax1.plot(running, label="Empirical coverage")
        ax1.axhline(1 - target_alpha, color="red", linestyle="--",
                    label=f"Target {1-target_alpha:.2f}")
        ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Coverage")
    ax1.set_title("DtACI Empirical Coverage and Interval Width")
    ax1.legend()
    if width_log:
        ax2.plot(width_log, color="C2")
    ax2.set_ylabel("Interval width")
    ax2.set_xlabel("Online step")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_reward_curve(history: Dict[str, List[float]],
                          path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for name, h in history.items():
        if "reward" in h:
            r = np.array(h["reward"])
            w = max(1, len(r) // 20)
            kernel = np.ones(w) / w
            smooth = np.convolve(r, kernel, mode="valid")
            ax.plot(smooth, label=name)
    ax.set_title("Training Reward Curve (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (= -total_cost)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
