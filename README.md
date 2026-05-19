# DtACI-Shielded DQN for Datacenter Compute-Power Co-optimisation

Research code implementing **DQN + Dynamic Tuned Adaptive Conformal Inference (DtACI)** for joint server scaling and task dispatch in datacenters under electricity market constraints.

---

## Project Structure

```
.
├── main.py                              # Unified CLI entry point (all baselines)
├── config.yaml                          # Single source of truth for all parameters
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Project metadata
├── _common.py                           # Shared experiment utilities
├── _heuristic_runner.py                 # Heuristic policy evaluation harness
├── 区域负荷2020-2023数据.csv             # Raw regional load data
│
├── src/
│   ├── __init__.py
│   ├── utils.py                         # Config loading, seeding, logging
│   ├── data_preprocess.py               # CSV → 15-min load + train/cal/test split
│   ├── datacenter_env.py                # Discrete-time datacenter environment
│   ├── price_model.py                   # Time-of-use electricity price curve
│   ├── workload_generator.py            # Synthetic task arrival generation
│   │
│   ├── conformal/
│   │   ├── aci.py                       # Adaptive Conformal Inference
│   │   ├── dtaci.py                     # Dynamic Tuned ACI (expert mixture)
│   │   ├── forecaster.py                # Lightweight time-series forecasters
│   │   └── split_conformal.py           # Standard split conformal prediction
│   │
│   ├── rl/
│   │   ├── dqn_agent.py                 # DQN agent (Q-network, replay, training)
│   │   ├── train_dqn.py                 # Generic training/eval loop + EpisodeStats
│   │   ├── episode_stats.py             # Per-episode stats accumulator
│   │   └── augmenters.py                # State augmenters (ACI/DtACI + shield)
│   │
│   ├── baselines/
│   │   ├── fixed_policy.py              # Fixed server count baseline
│   │   ├── queue_greedy_policy.py       # Queue-aware greedy baseline
│   │   └── price_aware_greedy_policy.py # Price-aware greedy (port of greedy_policy.m)
│   │
│   ├── safe_layer/
│   │   └── dtaci_action_shield.py       # DtACI action shield (safety layer)
│   │
│   └── evaluation/
│       ├── metrics.py                   # Aggregation: daily stats → summary
│       └── plot.py                      # Matplotlib figures for the report
│
├── tests/
│   ├── _fixtures.py                     # Test fixtures (tiny config, synthetic data)
│   ├── test_env.py                      # Environment unit tests
│   ├── test_dtaci.py                    # DtACI unit tests
│   └── test_shield.py                   # Action shield unit tests
│
└── outputs/                             # Generated at runtime
    ├── processed/                       # Cleaned load data and splits
    ├── logs/                            # Training/eval logs
    ├── figures/                         # Generated plots
    ├── daily_results.csv                # Per-method per-day metrics
    └── experiment_summary.csv           # Aggregated comparison table
```

---

## Baselines

| Method | Type | Description |
|--------|------|-------------|
| `fixed` | Heuristic | Constant server count (midpoint of Nmin..Nmax) |
| `queue_greedy` | Heuristic | Queue-aware greedy, ignores electricity price |
| `price_aware_greedy` | Heuristic | Price-aware greedy (defers P3 during high-price periods) |
| `dqn` | RL | Plain DQN (no conformal prediction, no shield) |
| `aci_dqn` | RL | DQN with state augmented by ACI prediction intervals |
| `dtaci_dqn` | RL | **Proposed**: DtACI-augmented state + DtACI action shield |

---

## Quick Start

### 1. Install dependencies

```bash
# Create venv (Python ≥3.11 recommended)
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Run experiments

```bash
# Run all 6 baselines end-to-end
python main.py --all

# Run only the proposed method (DtACI-DQN)
python main.py --method dtaci_dqn

# Run selected baselines
python main.py --method fixed queue_greedy dqn

# Skip preprocessing (use cached data)
python main.py --all --skip-preprocess

# Override random seed
python main.py --all --seed 42

# Skip plot generation
python main.py --all --no-plots

# Use custom config file
python main.py --all --config my_config.yaml
```

### 3. Run tests

```bash
python -m pytest tests/ -v
```

---

## Environment

The datacenter operates in discrete **T = 96 slots** (Δt = 15 min). Each slot, the agent selects the number of active servers n ∈ [Nmin, Nmax]. Three priority classes of tasks (P1 urgent, P2 interactive, P3 batch) arrive via Poisson processes whose rates are driven by a normalised regional load curve x(t).

Per-step cost:

$$c_t = \underbrace{E_t \cdot \pi_t}_{\text{electricity}} + \underbrace{\sum_k \beta_k v_{k,t} + \rho_k o_{k,t}}_{\text{SLA penalties}} + \underbrace{\phi \,|\Delta n_t|}_{\text{switching}}$$

where E_t is facility energy (kWh), π_t is time-of-use price (CNY/kWh), β_k/ρ_k are SLA violation and overdue penalties, and φ is the switching cost.

**State** (11-dim): [Q1, Q2, Q3, B1, B2, B3, price, x_load, n_prev, sin(2πt/T), cos(2πt/T)]

**Action**: discrete index ∈ {0..action_bins-1}, mapped linearly to [Nmin, Nmax]

**Reward**: r_t = −reward_scale × c_t

---

## DtACI Action Shield

Before an action reaches the environment, the safety layer computes conformal upper bounds on future P1+P2 arrivals (H-step look-ahead) using DtACI's expert mixture. If the proposed server count cannot guarantee covering the protected workload, the action is raised to the minimum safe level.

DtACI maintains a mixture of ACI experts with different learning rates (η), adapting online to distribution shifts while providing finite-sample marginal coverage guarantees.

---

## Key Config Parameters (`config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server.Nmin` / `Nmax` | 8 / 120 | Server count bounds |
| `server.target_util` | 0.75 | Target utilisation for greedy baselines |
| `server.ramp_limit` | 25 | Max server change per slot |
| `qos.deadline_slots` | [2, 8, 32] | Deadlines per priority (slots) |
| `qos.sla_penalty` | [20, 8, 2] | SLA violation penalty β_k |
| `price.high` / `middle` / `low` | 0.95 / 0.55 / 0.32 | TOU price tiers (CNY/kWh) |
| `conformal.alpha` | 0.10 | Target miscoverage rate |
| `conformal.horizon` | 4 | H-step look-ahead for shield |
| `conformal.dtaci_etas` | [0.005, 0.02, 0.05, 0.1, 0.25] | DtACI expert learning rates |
| `rl.train_episodes` | 300 | Training episodes |
| `rl.action_bins` | 21 | Action discretisation bins |
| `rl.hidden_sizes` | [128, 128] | DQN hidden layer sizes |
| `rl.gamma` | 0.99 | Discount factor |
| `seed` | 2024 | Global random seed |

---

## Outputs

- `outputs/processed/` — Cleaned 15-min load matrix and train/cal/test split
- `outputs/logs/` — Run logs (e.g. `run.log`, `dtaci_dqn_train.log`)
- `outputs/figures/` — Bar charts, SLA comparison, training curves
- `outputs/daily_results.csv` — Per-method per-day detailed metrics
- `outputs/experiment_summary.csv` — Aggregated comparison across methods

---

## Reproducibility

All random seeds, hyperparameters, and data splits are fixed in `config.yaml`. Running `python main.py --all` will reproduce the results within floating-point non-determinism across hardware.

---

## References

This implementation corresponds to:

> *Adaptive Conformal Inference-Shielded Deep Q-Network for Compute-Power
> Co-optimisation in Electricity-Market-Aware Datacenters* (2024).

The regional load data (`区域负荷2020-2023数据.csv`) is used **only** as a shape proxy for generating synthetic workloads and does not represent real datacenter traces.
