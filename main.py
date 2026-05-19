"""
main.py -- Unified experiment runner for datacenter compute-power co-optimisation.

Runs one or all baseline policies and reports results.

Baselines
---------
  Heuristic (no training):
    fixed              Fixed server count (midpoint of Nmin..Nmax)
    queue_greedy       Queue-aware greedy, ignores electricity price
    price_aware_greedy Price-aware greedy (port of greedy_policy.m)

  RL (requires PyTorch):
    dqn                Plain DQN, no conformal prediction, no safety shield
    aci_dqn            ACI-DQN: state augmented with ACI prediction intervals
    dtaci_dqn          DtACI-DQN: DtACI intervals + action shield (PROPOSED)

Usage
-----
    python main.py --all                          # run all 6 baselines
    python main.py --method dtaci_dqn             # run only the proposed method
    python main.py --method fixed queue_greedy    # run selected baselines
    python main.py --all --skip-preprocess        # skip data preprocessing
    python main.py --all --config my_config.yaml  # use custom config
    python main.py --all --seed 42                # override random seed
    python main.py --all --no-plots               # skip figure generation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `src` and `_common` are importable.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.utils import ensure_dir, get_logger, load_config, set_global_seed
from src.data_preprocess import run as preprocess_run
from src.datacenter_env import DataCenterEnv
from src.rl.dqn_agent import DQNAgent
from src.rl.train_dqn import (
    EpisodeStats,
    IdentityAugmenter,
    evaluate,
    train_agent,
)
from src.rl.augmenters import ConformalAugmenter
from src.baselines.fixed_policy import FixedPolicy
from src.baselines.queue_greedy_policy import QueueGreedyPolicy
from src.baselines.price_aware_greedy_policy import PriceAwareGreedyPolicy
from src.evaluation.metrics import aggregate_to_summary, episode_stats_to_row
from src.evaluation.plot import (
    bar_compare,
    grouped_sla_bar,
    training_reward_curve,
)

from _common import build_env_and_splits, calibration_arrival_data
from _heuristic_runner import run_heuristic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_action_debug(method: str, action_logs: list, cfg: Dict,
                        env) -> None:
    """Write per-slot action diagnostics for a few test days."""
    rows = []
    for day_idx, day_log in action_logs:
        for slot, (raw, safe, n_active) in enumerate(day_log):
            rows.append({
                "method": method,
                "day_index": day_idx,
                "slot": slot,
                "raw_action": raw,
                "shielded_action": safe,
                "mapped_servers": n_active,
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["outputs_dir"])
    df.to_csv(out_dir / f"action_debug_{method}.csv", index=False)


def _write_action_distribution(daily_df: pd.DataFrame, cfg: Dict) -> None:
    """Aggregate server distribution across all methods from daily results."""
    rows = []
    for method in daily_df["method"].unique():
        method_data = daily_df[daily_df["method"] == method]
        for _, row in method_data.iterrows():
            rows.append({
                "method": method,
                "avg_active_servers": row["average_active_servers"],
            })
    if not rows:
        return
    dist_df = pd.DataFrame(rows)
    # Compute distribution statistics per method.
    stats_rows = []
    for method in dist_df["method"].unique():
        vals = dist_df[dist_df["method"] == method]["avg_active_servers"]
        stats_rows.append({
            "method": method,
            "mean_servers": float(vals.mean()),
            "std_servers": float(vals.std()),
            "min_servers": float(vals.min()),
            "max_servers": float(vals.max()),
            "n_days": len(vals),
        })
    dist_summary = pd.DataFrame(stats_rows)
    out_dir = Path(cfg["paths"]["outputs_dir"])
    dist_summary.to_csv(out_dir / "action_distribution.csv", index=False)


# ---------------------------------------------------------------------------
# Registry of all available methods
# ---------------------------------------------------------------------------

HEURISTIC_METHODS = ["fixed", "queue_greedy", "price_aware_greedy"]
RL_METHODS = ["dqn", "aci_dqn", "dtaci_dqn"]
ALL_METHODS = HEURISTIC_METHODS + RL_METHODS


# ---------------------------------------------------------------------------
# Heuristic runners
# ---------------------------------------------------------------------------

def _make_policy(method: str, cfg: Dict):
    if method == "fixed":
        return FixedPolicy(cfg)
    elif method == "queue_greedy":
        return QueueGreedyPolicy(cfg)
    elif method == "price_aware_greedy":
        return PriceAwareGreedyPolicy(cfg)
    else:
        raise ValueError(f"Unknown heuristic method: {method}")


def run_heuristic_baseline(method: str, cfg: Dict, env, splits, rng,
                           log) -> Dict:
    log.info(f"  Running {method} on {len(splits['test'])} test days ...")
    policy = _make_policy(method, cfg)
    stats = run_heuristic(env, policy, splits["test"], rng)
    rows = [episode_stats_to_row(method, s) for s in stats]
    return {"stats": stats, "rows": rows, "extra_metrics": {}}


# ---------------------------------------------------------------------------
# RL runners
# ---------------------------------------------------------------------------

def _build_rl_agent(method: str, cfg: Dict, env):
    """Build DQNAgent and augmenter for a given RL method."""
    rl_cfg = cfg["rl"]

    if method == "dqn":
        aug = IdentityAugmenter()
        state_dim = env.observation_dim
    elif method in ("aci_dqn", "dtaci_dqn"):
        learner = "aci" if method == "aci_dqn" else "dtaci"
        use_shield = (method == "dtaci_dqn")
        aug = ConformalAugmenter(cfg, learner=learner, use_shield=use_shield)
        state_dim = env.observation_dim + 2 * cfg["qos"]["K"]
    else:
        raise ValueError(f"Unknown RL method: {method}")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=env.action_dim,
        hidden=rl_cfg["hidden_sizes"],
        lr=rl_cfg["lr"],
        gamma=rl_cfg["gamma"],
        batch_size=rl_cfg["batch_size"],
        replay_size=rl_cfg["replay_size"],
        epsilon_start=rl_cfg["epsilon_start"],
        epsilon_end=rl_cfg["epsilon_end"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        target_update_interval=rl_cfg["target_update_interval"],
        learning_starts=rl_cfg["learning_starts"],
        reward_scale=rl_cfg["reward_scale"],
        max_grad_norm=rl_cfg["max_grad_norm"],
    )
    return agent, aug


def run_rl_baseline(method: str, cfg: Dict, env, splits, norm_matrix,
                    seed: int, log) -> Dict:
    rl_cfg = cfg["rl"]

    agent, aug = _build_rl_agent(method, cfg, env)

    # Warm up conformal learners on calibration data
    if method in ("aci_dqn", "dtaci_dqn"):
        y_hat_cal, y_cal = calibration_arrival_data(
            cfg, norm_matrix, splits["calibration"]
        )
        aug.cp.warm_up_from_calibration(y_hat_cal, y_cal)
        log.info(f"  Warmed up {method} conformal learners on "
                 f"{len(splits['calibration'])} calibration days.")

    rng = np.random.default_rng(seed)

    # Train
    log.info(f"  Training {method} for {rl_cfg['train_episodes']} episodes ...")
    log_path = str(
        Path(cfg["paths"]["logs_dir"]) / f"{method}_train.log"
    )
    history = train_agent(
        env, agent, aug, splits["train"],
        episodes=rl_cfg["train_episodes"],
        rng=rng, log_path=log_path,
    )

    # Save model
    models_dir = ensure_dir(cfg["paths"]["models_dir"])
    model_path = str(models_dir / f"{method}.pt")
    agent.save(model_path)
    log.info(f"  Saved model to {model_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = Path(cfg["paths"]["outputs_dir"]) / f"{method}_training_history.csv"
    history_df.to_csv(history_path, index=False)
    log.info(f"  Training history -> {history_path}")

    # Evaluate with epsilon=0 (pure greedy)
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    eval_days = splits["test"]
    log.info(f"  Evaluating {method} on {len(eval_days)} test days ...")
    eval_rng = np.random.default_rng(seed + 30)
    # Record detailed actions for first 3 test days for diagnostics.
    debug_days = eval_days[:3]
    _, action_logs = evaluate(env, agent, aug, debug_days,
                              rng=eval_rng, record_actions=True)
    # Evaluate fully on all test days.
    stats = evaluate(env, agent, aug, eval_days, rng=eval_rng, record_actions=False)
    agent.epsilon = old_eps

    # Write action debug CSV for diagnostics.
    _write_action_debug(method, action_logs, cfg, env)

    rows = [episode_stats_to_row(method, s) for s in stats]
    extra = aug.metrics() if hasattr(aug, "metrics") else {}
    return {
        "stats": stats,
        "rows": rows,
        "history": history,
        "extra_metrics": extra,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiments(cfg: Dict, methods: List[str], seed: int,
                    skip_preprocess: bool = False,
                    skip_plots: bool = False,
                    log=None) -> None:
    log = log or get_logger("main")
    set_global_seed(seed)

    t_start = time.time()

    # ---- Stage 1: Preprocess ------------------------------------------------
    if not skip_preprocess:
        log.info("=" * 60)
        log.info("STAGE: Data preprocessing")
        log.info("=" * 60)
        preprocess_run(cfg)

    # ---- Stage 2: Build env and splits --------------------------------------
    log.info("=" * 60)
    log.info("STAGE: Building environment and data splits")
    log.info("=" * 60)
    env, splits, norm_matrix = build_env_and_splits(cfg)
    log.info(f"  Train days: {len(splits['train'])}")
    log.info(f"  Calibration days: {len(splits['calibration'])}")
    log.info(f"  Test days: {len(splits['test'])}")

    all_rows: List[Dict] = []
    all_histories: Dict[str, Dict] = {}
    all_extra: Dict[str, Dict] = {}

    # ---- Stage 3: Run each method -------------------------------------------
    for method in methods:
        log.info("=" * 60)
        log.info(f"METHOD: {method}")
        log.info("=" * 60)

        if method in HEURISTIC_METHODS:
            rng = np.random.default_rng(seed)
            result = run_heuristic_baseline(method, cfg, env, splits, rng, log)
            all_rows.extend(result["rows"])

        elif method in RL_METHODS:
            rl_seed = seed + {"dqn": 3, "aci_dqn": 4, "dtaci_dqn": 5}[method]
            result = run_rl_baseline(
                method, cfg, env, splits, norm_matrix, rl_seed, log
            )
            all_rows.extend(result["rows"])
            if "history" in result:
                all_histories[method] = result["history"]
            if result.get("extra_metrics"):
                all_extra[method] = result["extra_metrics"]

        else:
            log.warning(f"  Unknown method '{method}', skipping.")

    # ---- Stage 4: Summary ---------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE: Summary report")
    log.info("=" * 60)

    out_dir = ensure_dir(cfg["paths"]["outputs_dir"])
    daily_df = pd.DataFrame(all_rows)

    # Add metadata columns.
    test_days = splits["test"]
    daily_df["num_eval_days"] = len(test_days)
    daily_df["start_day_index"] = test_days[0]
    daily_df["end_day_index"] = test_days[-1]
    daily_df["random_seed"] = seed

    # Per-method daily results
    daily_path = out_dir / "daily_results.csv"
    daily_df.to_csv(daily_path, index=False)
    log.info(f"  Daily results -> {daily_path}")

    # Aggregate summary
    summary = aggregate_to_summary(daily_df, all_extra if all_extra else None)
    # Add metadata to summary.
    summary["num_eval_days"] = len(test_days)
    summary["start_day_index"] = test_days[0]
    summary["end_day_index"] = test_days[-1]
    summary["random_seed"] = seed
    summary_path = out_dir / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"  Experiment summary -> {summary_path}")

    # Action distribution
    _write_action_distribution(daily_df, cfg)
    log.info(f"  Action distribution -> {out_dir / 'action_distribution.csv'}")

    # Print comparison
    key_cols = ["method", "total_objective_cost", "electricity_cost",
                "qos_cost", "P1_sla_violation_rate", "average_utilization"]
    available = [c for c in key_cols if c in summary.columns]
    log.info("\n" + summary[available].to_string())

    # ---- Stage 5: Plots -----------------------------------------------------
    if not skip_plots:
        log.info("=" * 60)
        log.info("STAGE: Generating figures")
        log.info("=" * 60)
        fig_dir = ensure_dir(cfg["paths"]["figures_dir"])

        try:
            bar_compare(
                summary, "total_objective_cost",
                "Average Total Cost by Method", "Total cost (CNY)",
                fig_dir / "bar_total_cost.png",
            )
            bar_compare(
                summary, "P1_sla_violation_rate",
                "P1 SLA Violation Rate by Method", "Violation rate",
                fig_dir / "bar_p1_violation.png",
            )
            if all(c in daily_df.columns for c in
                   ["P1_sla_violation_rate", "P2_sla_violation_rate",
                    "P3_sla_violation_rate"]):
                grouped_sla_bar(summary, fig_dir / "grouped_sla.png")

            if all_histories:
                training_reward_curve(
                    all_histories, fig_dir / "training_reward.png"
                )

            log.info(f"  Figures saved to {fig_dir}")
        except Exception as e:
            log.warning(f"  Plotting failed: {e}")

    elapsed = time.time() - t_start
    log.info(f"\nAll done in {elapsed:.1f}s.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Datacenter DtACI-DQN experiment runner."
    )
    parser.add_argument(
        "--method", nargs="+", choices=ALL_METHODS,
        help="Which baseline(s) to run.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 6 baselines.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config.",
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip data preprocessing (use cached processed data).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all and not args.method:
        print("Please specify --method or --all. Use --help for usage.")
        sys.exit(1)

    methods = ALL_METHODS if args.all else args.method

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 2024))

    log_dir = ensure_dir(cfg["paths"]["logs_dir"])
    log = get_logger("main", log_file=str(log_dir / "run.log"))

    log.info(f"Config: {args.config}")
    log.info(f"Methods: {methods}")
    log.info(f"Seed: {seed}")

    run_experiments(
        cfg=cfg,
        methods=methods,
        seed=seed,
        skip_preprocess=args.skip_preprocess,
        skip_plots=args.no_plots,
        log=log,
    )


if __name__ == "__main__":
    main()
