"""
main.py -- Unified experiment runner for datacenter compute-power co-optimisation.

Runs one or all baseline policies across E0-E4 scenarios and reports results.

Main methods (9)
---------------
  Heuristic:
    fixed               Fixed server count (midpoint of Nmin..Nmax)
    queue_greedy        Queue-aware greedy, ignores electricity price
    price_aware_greedy  Price-aware greedy (port of greedy_policy.m)
    forecast_greedy     Rolling-mean point forecast + capacity planning
    conformal_greedy    ACI upper-bound forecast + conservative capacity planning

  RL (requires PyTorch):
    dqn                    Plain DQN (17-dim state)
    forecast_dqn           DQN + rolling-mean point-forecast features (20-dim)
    static_conformal_dqn   DQN + fixed split-conformal interval features (23-dim)
    aci_dqn                DQN + ACI online adaptive interval features (23-dim)

Appendix:
    shielded_dtaci_dqn     DtACI-DQN with action shield (appendix only)

Scenarios: E0 Easy, E1 Normal-Hard, E2 Distribution Shift, E3 Bursty-Uncertainty,
           E4 Capacity Cliff.

Usage
-----
    python main.py --all --scenario E1
    python main.py --method dqn aci_dqn --scenario E1
    python main.py --all --scenario E1 --skip-plots
    python main.py --method dqn --scenario E1 --training-seeds 2024 --scenario-seeds 100 101
"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
    rollout_episode,
    train_agent,
)
from src.rl.augmenters import (
    ConformalAugmenter,
    ForecastAugmenter,
    StaticConformalAugmenter,
)
from src.scenarios import (
    build_scenario_config,
    get_phase_config,
    apply_scenario_to_env,
    list_available_scenarios,
)
from src.baselines.fixed_policy import FixedPolicy
from src.baselines.queue_greedy_policy import QueueGreedyPolicy
from src.baselines.price_aware_greedy_policy import PriceAwareGreedyPolicy
from src.baselines.forecast_greedy_policy import ForecastGreedyPolicy
from src.baselines.conformal_greedy_policy import ConformalGreedyPolicy
from src.evaluation.metrics import (
    aggregate_to_summary,
    aggregate_with_ci,
    compute_paired_tests,
    episode_stats_to_row,
)
from src.evaluation.plot import (
    bar_compare,
    grouped_sla_bar,
    training_reward_curve,
)

from _common import build_env_and_splits, calibration_arrival_data
from _heuristic_runner import run_heuristic

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

MAIN_HEURISTIC_METHODS = [
    "fixed", "queue_greedy", "price_aware_greedy",
    "forecast_greedy", "conformal_greedy",
]
MAIN_RL_METHODS = [
    "dqn", "forecast_dqn", "static_conformal_dqn", "aci_dqn",
]
ALL_MAIN_METHODS = MAIN_HEURISTIC_METHODS + MAIN_RL_METHODS
APPENDIX_METHODS = ["shielded_dtaci_dqn"]


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def _make_policy(method: str, cfg: Dict):
    if method == "fixed":
        return FixedPolicy(cfg)
    elif method == "queue_greedy":
        return QueueGreedyPolicy(cfg)
    elif method == "price_aware_greedy":
        return PriceAwareGreedyPolicy(cfg)
    elif method == "forecast_greedy":
        return ForecastGreedyPolicy(cfg)
    elif method == "conformal_greedy":
        return ConformalGreedyPolicy(cfg)
    else:
        raise ValueError(f"Unknown heuristic method: {method}")


# ---------------------------------------------------------------------------
# RL agent factory
# ---------------------------------------------------------------------------

def _build_rl_agent(method: str, cfg: Dict, env: DataCenterEnv):
    rl_cfg = cfg["rl"]
    K = cfg["qos"]["K"]

    if method == "dqn":
        aug = IdentityAugmenter()
        state_dim = env.observation_dim
    elif method == "forecast_dqn":
        aug = ForecastAugmenter(cfg)
        state_dim = env.observation_dim + K          # 17 + 3 = 20
    elif method == "static_conformal_dqn":
        aug = StaticConformalAugmenter(cfg)
        state_dim = env.observation_dim + 2 * K      # 17 + 6 = 23
    elif method == "aci_dqn":
        aug = ConformalAugmenter(cfg, learner="aci", use_shield=False)
        state_dim = env.observation_dim + 2 * K      # 17 + 6 = 23
    elif method == "shielded_dtaci_dqn":
        aug = ConformalAugmenter(cfg, learner="dtaci", use_shield=True)
        state_dim = env.observation_dim + 2 * K
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


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_heuristic_method(method: str, cfg: Dict, env, splits, rng,
                         log) -> Dict:
    """Run a heuristic method on test days. Returns {stats, rows, extra_metrics}."""
    log.info(f"  Running {method} on {len(splits['test'])} test days ...")
    policy = _make_policy(method, cfg)

    # Conformal-Greedy needs calibration warm-up
    if method == "conformal_greedy" and hasattr(policy, "warm_up_from_calibration"):
        norm = env.day_norm_matrix
        y_hat_cal, y_cal = calibration_arrival_data(cfg, norm, splits["calibration"])
        policy.warm_up_from_calibration(y_hat_cal, y_cal)
        log.info(f"  Warmed up {method} on {len(splits['calibration'])} calibration days.")

    stats = run_heuristic(env, policy, splits["test"], rng)
    rows = [episode_stats_to_row(method, s) for s in stats]
    extra = policy.metrics() if hasattr(policy, "metrics") else {}
    return {"stats": stats, "rows": rows, "extra_metrics": extra}


def run_rl_method(method: str, cfg: Dict, env, splits, norm_matrix,
                  seed: int, log) -> Dict:
    """Run one RL method (train + eval)."""
    rl_cfg = cfg["rl"]
    agent, aug = _build_rl_agent(method, cfg, env)

    # Warm up conformal learners on calibration data
    needs_warmup = method in (
        "aci_dqn", "static_conformal_dqn", "shielded_dtaci_dqn",
    )
    if needs_warmup:
        y_hat_cal, y_cal = calibration_arrival_data(
            cfg, norm_matrix, splits["calibration"],
        )
        aug.warm_up_from_calibration(y_hat_cal, y_cal)
        log.info(f"  Warmed up {method} on {len(splits['calibration'])} calibration days.")

    rng = np.random.default_rng(seed)

    # Train
    log.info(f"  Training {method} for {rl_cfg['train_episodes']} episodes ...")
    log_path = str(Path(cfg["paths"]["logs_dir"]) / f"{method}_train.log")
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

    # Evaluate
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    eval_days = splits["test"]
    log.info(f"  Evaluating {method} on {len(eval_days)} test days ...")
    eval_rng = np.random.default_rng(seed + 30)
    stats = evaluate(env, agent, aug, eval_days, rng=eval_rng, record_actions=False)
    agent.epsilon = old_eps

    rows = [episode_stats_to_row(method, s) for s in stats]
    extra = aug.metrics() if hasattr(aug, "metrics") else {}
    return {
        "stats": stats,
        "rows": rows,
        "history": history,
        "extra_metrics": extra,
    }


# ---------------------------------------------------------------------------
# ACI diagnostics collection (in-rollout recording)
# ---------------------------------------------------------------------------

def collect_aci_diagnostics(env: DataCenterEnv,
                            agent,
                            aug,
                            day_indices: List[int],
                            rng) -> pd.DataFrame:
    """Roll out with diagnostics recording for ACI methods.

    Returns a DataFrame with columns:
    [method, day_index, slot, priority, alpha, interval_lo, interval_hi,
     y_hat, y_true, coverage, residual_quantile, n_active]
    """
    rows = []
    method = aug.name if hasattr(aug, "name") else "unknown"
    for d in day_indices:
        state, _ = env.reset(int(d))
        aug.reset(env, int(d))
        while not env.done:
            s_aug = aug.augment(state, env)
            a_raw = (agent.select_action(s_aug, greedy=True)
                     if agent is not None else 0)
            from src.datacenter_env import action_to_n, n_to_action
            a_n = action_to_n(a_raw, env.cfg)
            a_safe = aug.shield(a_n, s_aug, env)
            a_safe_idx = n_to_action(a_safe, env.cfg)
            next_state, _, _, info = env.step(a_safe_idx)
            aug.on_step(env, info, a_raw, a_safe)

            # Record step diagnostics if augmenter supports it
            if hasattr(aug, "_step_diagnostics_history"):
                for diag in aug._step_diagnostics_history[-1:]:
                    for k in range(env.K):
                        rows.append({
                            "method": method,
                            "day_index": int(d),
                            "slot": env.t - 1,
                            "priority": k + 1,
                            "alpha": diag.get("alpha", [0.0] * env.K)[k],
                            "interval_lo": diag.get("interval_lo", np.zeros((env.K, 1)))[k, 0],
                            "interval_hi": diag.get("interval_hi", np.zeros((env.K, 1)))[k, 0],
                            "coverage": diag.get("coverage", [0.0] * env.K)[k],
                            "residual_quantile": diag.get("residual_quantile", [0.0] * env.K)[k],
                            "n_active": diag.get("n_active", 0),
                        })
            state = next_state
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiments(cfg: Dict,
                    methods: List[str],
                    scenario_id: str,
                    training_seeds: List[int],
                    scenario_seeds: List[int],
                    skip_preprocess: bool = False,
                    skip_plots: bool = False,
                    log=None) -> None:
    log = log or get_logger("main")

    t_start = time.time()

    # ---- Stage 1: Preprocess ------------------------------------------------
    if not skip_preprocess:
        log.info("=" * 60)
        log.info("STAGE: Data preprocessing")
        log.info("=" * 60)
        preprocess_run(cfg)

    # ---- Stage 2: Build base env and splits ---------------------------------
    log.info("=" * 60)
    log.info("STAGE: Building environment and data splits")
    log.info("=" * 60)
    base_env, splits, norm_matrix = build_env_and_splits(cfg)
    log.info(f"  Train days: {len(splits['train'])}")
    log.info(f"  Calibration days: {len(splits['calibration'])}")
    log.info(f"  Test days: {len(splits['test'])}")

    out_dir = ensure_dir(cfg["paths"]["outputs_dir"])
    all_daily_rows: List[Dict] = []
    all_diag_dfs: List[pd.DataFrame] = []

    # ---- Stage 3: Loop over scenario seeds ----------------------------------
    for s_seed in scenario_seeds:
        log.info("=" * 60)
        log.info(f"SCENARIO SEED: {s_seed}  |  Scenario: {scenario_id}")
        log.info("=" * 60)

        scenario_cfg = build_scenario_config(scenario_id, cfg)
        set_global_seed(s_seed)

        for method in methods:
            log.info("-" * 40)
            log.info(f"METHOD: {method}")

            if method in APPENDIX_METHODS:
                log.info(f"  (appendix method, skipping in main run)")
                continue

            if method in MAIN_HEURISTIC_METHODS:
                # Heuristics: one run per scenario_seed, training_seed = None
                # Use the phase config appropriate for evaluation (test phase)
                test_cfg = get_phase_config(scenario_cfg, "test")
                env = deepcopy(base_env)
                # Re-init with scenario cfg for server/power params
                env.cfg = test_cfg
                env.Nmin = int(test_cfg["server"]["Nmin"])
                env.Nmax = int(test_cfg["server"]["Nmax"])
                env.cap = float(test_cfg["server"]["cap_per_server"])
                env.ramp = int(test_cfg["server"]["ramp_limit"])
                env.c_sw = float(test_cfg["power"]["switch_cost"])
                apply_scenario_to_env(env, test_cfg)

                result = run_heuristic_method(
                    method, test_cfg, env, splits, None, log,
                )
                for row in result["rows"]:
                    row["scenario_id"] = scenario_id
                    row["scenario_seed"] = s_seed
                    row["training_seed"] = None
                all_daily_rows.extend(result["rows"])

            elif method in MAIN_RL_METHODS or method in APPENDIX_METHODS:
                for t_seed in training_seeds:
                    log.info(f"  Training seed: {t_seed}")
                    set_global_seed(t_seed)

                    # Use train phase config for training, test for eval
                    train_cfg = get_phase_config(scenario_cfg, "train")
                    test_cfg = get_phase_config(scenario_cfg, "test")

                    env = deepcopy(base_env)
                    env.cfg = train_cfg  # training uses train config
                    env.Nmin = int(train_cfg["server"]["Nmin"])
                    env.Nmax = int(train_cfg["server"]["Nmax"])
                    env.cap = float(train_cfg["server"]["cap_per_server"])
                    env.c_sw = float(train_cfg["power"]["switch_cost"])
                    apply_scenario_to_env(env, train_cfg)

                    result = run_rl_method(
                        method, train_cfg, env, splits, norm_matrix, t_seed, log,
                    )
                    for row in result["rows"]:
                        row["scenario_id"] = scenario_id
                        row["scenario_seed"] = s_seed
                        row["training_seed"] = t_seed
                    all_daily_rows.extend(result["rows"])

                    # ACI diagnostics (only for methods with conformal augmenters)
                    if method in ("aci_dqn", "shielded_dtaci_dqn"):
                        agent2, aug2 = _build_rl_agent(method, train_cfg, env)
                        # Load trained model
                        agent2.load(str(Path(cfg["paths"]["models_dir"]) / f"{method}.pt"))
                        agent2.epsilon = 0.0
                        diag_df = collect_aci_diagnostics(
                            env, agent2, aug2, splits["test"][:5], None,
                        )
                        diag_df["scenario_id"] = scenario_id
                        diag_df["scenario_seed"] = s_seed
                        diag_df["training_seed"] = t_seed
                        all_diag_dfs.append(diag_df)
            else:
                log.warning(f"  Unknown method '{method}', skipping.")

    # ---- Stage 4: Summary ---------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE: Summary report")
    log.info("=" * 60)

    daily_df = pd.DataFrame(all_daily_rows)

    # Per-method daily results
    daily_path = out_dir / "daily_results.csv"
    daily_df.to_csv(daily_path, index=False)
    log.info(f"  Daily results -> {daily_path}")

    # Aggregate with CI
    summary = aggregate_with_ci(daily_df, group_cols=["method"])
    summary_path = out_dir / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"  Experiment summary -> {summary_path}")

    # Paired comparison tests
    paired = compute_paired_tests(daily_df, group_col="method")
    paired_path = out_dir / "paired_tests.csv"
    paired.to_csv(paired_path, index=False)
    log.info(f"  Paired tests -> {paired_path}")

    # ACI diagnostics
    if all_diag_dfs:
        diag_all = pd.concat(all_diag_dfs, ignore_index=True)
        diag_path = out_dir / "aci_diagnostics.csv"
        diag_all.to_csv(diag_path, index=False)
        log.info(f"  ACI diagnostics -> {diag_path}")

    # Print comparison
    key_cols = ["method", "mean_cost", "std_cost", "ci_lower", "ci_upper",
                "mean_utilization", "mean_p1_violation"]
    available = [c for c in key_cols if c in summary.columns]
    if available:
        log.info("\n" + summary[available].to_string())

    # ---- Stage 5: Plots -----------------------------------------------------
    if not skip_plots:
        log.info("=" * 60)
        log.info("STAGE: Generating figures")
        log.info("=" * 60)
        fig_dir = ensure_dir(cfg["paths"]["figures_dir"])
        try:
            if "mean_cost" in summary.columns:
                bar_compare(
                    summary, "mean_cost",
                    "Average Total Cost by Method", "Total cost (CNY)",
                    fig_dir / "bar_total_cost.png",
                )
        except Exception as e:
            log.warning(f"  Plotting failed: {e}")

    elapsed = time.time() - t_start
    log.info(f"\nAll done in {elapsed:.1f}s.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Datacenter ACI-DQN experiment runner."
    )
    parser.add_argument(
        "--method", nargs="+", choices=ALL_MAIN_METHODS + APPENDIX_METHODS,
        help="Which method(s) to run.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 9 main methods.",
    )
    parser.add_argument(
        "--scenario", type=str, default="E1",
        choices=list_available_scenarios(),
        help="Scenario ID (default: E1 Normal-Hard).",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--training-seeds", type=int, nargs="+", default=None,
        help="Training seeds for RL methods (default: from config).",
    )
    parser.add_argument(
        "--scenario-seeds", type=int, nargs="+", default=None,
        help="Scenario seeds (default: from config).",
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip data preprocessing.",
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

    methods = ALL_MAIN_METHODS if args.all else args.method

    cfg = load_config(args.config)
    scenario_id = args.scenario

    training_seeds = (args.training_seeds
                      if args.training_seeds is not None
                      else cfg.get("experiment", {}).get("training_seeds", [2024]))
    scenario_seeds = (args.scenario_seeds
                      if args.scenario_seeds is not None
                      else cfg.get("experiment", {}).get("scenario_seeds", [100]))

    log_dir = ensure_dir(cfg["paths"]["logs_dir"])
    log = get_logger("main", log_file=str(log_dir / "run.log"))

    log.info(f"Config: {args.config}")
    log.info(f"Scenario: {scenario_id}")
    log.info(f"Methods: {methods}")
    log.info(f"Training seeds: {training_seeds}")
    log.info(f"Scenario seeds: {scenario_seeds}")

    run_experiments(
        cfg=cfg,
        methods=methods,
        scenario_id=scenario_id,
        training_seeds=training_seeds,
        scenario_seeds=scenario_seeds,
        skip_preprocess=args.skip_preprocess,
        skip_plots=args.no_plots,
        log=log,
    )


if __name__ == "__main__":
    main()
