# ACI-DQN: Adaptive Conformal Inference for Deep RL Datacenter Scheduling

> Under high-pressure, distribution-shift, and burst-uncertainty scenarios, **ACI-DQN** uses **Adaptive Conformal Inference (ACI)** to augment the DQN state space, enabling robust dynamic server scheduling.

---

## Core Problem

Datacenters face the dual challenge of **heterogeneous task mixing** (urgent P1, interactive P2, batch P3) and **time-of-use electricity pricing**. The scheduler must decide how many servers to keep active every 15 minutes — too few means SLA violations (costly), too many means wasted electricity.

**Key difficulty**: Task arrival patterns exhibit **distribution shift** between training and deployment. Poisson arrival parameters that held during training break under seasonal peaks, bursty loads, and capacity cliffs. A standard DQN either over-provisions servers or racks up violations.

**This project proposes ACI-DQN**: infusing online uncertainty quantification from adaptive conformal inference into the DQN state space, allowing the agent to make precise scheduling decisions under distribution shift.

---

## Method Lineup

### 9 Main Methods

| Type       | Method                 | State Dim | Description                                              |
| ---------- | ---------------------- | --------- | -------------------------------------------------------- |
| Heuristic  | Fixed                  | —         | Constant server count, no adaptation                     |
| Heuristic  | Queue-Greedy           | —         | Serve current backlog, ignore price                      |
| Heuristic  | Price-Aware Greedy     | —         | Defer P3 when electricity is expensive                   |
| Heuristic  | Forecast-Greedy        | —         | Rolling-mean point forecast + greedy capacity planning   |
| Heuristic  | Conformal-Greedy       | —         | ACI interval forecast + greedy capacity planning         |
| RL         | DQN                    | 17        | Standard DQN, no uncertainty features                    |
| RL         | Forecast-DQN           | 20        | DQN + K point-forecast features (rolling mean)           |
| RL         | Static-Conformal-DQN   | 23        | DQN + 2K fixed split-conformal interval features         |
| **RL**     | **ACI-DQN**            | **23**    | **Proposed**: DQN + 2K ACI online-adaptive interval features |

### Appendix

| Method              | Description                                     |
| ------------------- | ----------------------------------------------- |
| Shielded DtACI-DQN  | DtACI meta-learning + safety action shield      |

---

## ACI-DQN Core Mechanism

```
Each decision step t:
┌───────────────────────────────────────────────┐
│  1. Rolling-mean forecaster: ŷ_k ← mean(past arrivals_k)  │
│  2. ACI online update: [L_k, U_k] ← adaptive interval     │
│     Algorithm: α_{t+1} = α_t + η·(α_target - 𝟙{covered})  │
│     Effect: arrival pattern shift → interval auto-widens   │
│  3. State concatenation: s' = [s_raw(17) ‖ Ū/λ ‖ L̄/λ]    │
│  4. Q-network inference: a = argmax Q(s', ·)               │
│  5. Env execution: n = action_to_n(a)                      │
└───────────────────────────────────────────────┘
```

**Key difference from DQN**: DQN only sees "how long is the queue now"; ACI-DQN additionally sees "how wide is the prediction interval for future arrivals" — wide intervals signal high uncertainty, prompting the network to allocate more resources proactively; narrow intervals signal predictability, allowing the network to scale down and save electricity.

All conformal methods (Conformal-Greedy, Static-Conformal-DQN, ACI-DQN) share the same rolling-mean base forecaster. The only difference is how intervals are constructed: static split-conformal (one-time calibration) vs. ACI (online adaptive).

---

## Experimental Setup

### Scenario Framework (E0–E4)

Experiments use a **config-isolated scenario overlay** system. Base `config.yaml` holds system defaults (Nmax=120, ramp_limit=25, switch_cost=0.10). Each scenario YAML in `configs/scenarios/` overlays capacity constraints and workload enhancements without mutating the base config.

| Scenario | Name                 | Nmax | Ramp | Switch | Key Challenge                        |
| -------- | -------------------- | ---- | ---- | ------ | ------------------------------------ |
| E0       | Easy                 | 120  | 25   | 0.10   | Sanity check, ample capacity         |
| E1       | Normal-Hard          | 75   | 15   | 0.30   | Main benchmark, tighter capacity     |
| E2       | Distribution Shift   | 75   | 15   | 0.30   | Train/cal normal, test shifted       |
| E3       | Bursty Uncertainty   | 75   | 15   | 0.30   | Baseline noise + P3 bursts at test   |
| E4       | Capacity Cliff       | 45   | 10   | 0.40   | Severe capacity pressure             |

Each scenario defines **phase-specific overrides** (`phases.train`, `phases.calibration`, `phases.test`) — train and calibration phases can use different distributions from test.

### Data & Splits

Based on real regional power load curves driving synthetic task generation, split chronologically:

| Set          | Days  | Purpose                               |
| ------------ | ----- | ------------------------------------- |
| Train        | 60%   | Train DQN / Forecast-DQN / Static-C-DQN / ACI-DQN |
| Calibration  | 20%   | Warm up conformal forecasters         |
| Test         | 20%   | Final evaluation (same deterministic trace per day) |

### Task Model

```
λ_P1(t) = 1.0 + 6.0·x(t) + 2.0·evening(t)
λ_P2(t) = 2.0 + 4.0·x(t) + 2.0·business(t)
λ_P3(t) = 0.5 + 3.5·night(t) + 2.0·(1-x(t))
N_k(t) ~ Poisson(λ_k(t)),  work ~ Normal(μ_k, σ_k)
```

| Priority | Deadline        | Mean Work | SLA Penalty   |
| -------- | --------------- | --------- | ------------- |
| P1       | 2 slots (30min) | 0.60      | 50.0 CNY/viol |
| P2       | 8 slots (2h)    | 1.20      | 20.0 CNY/viol |
| P3       | 32 slots (8h)   | 4.00      | 5.0 CNY/viol  |

### Workload Enhancements

Configurable via `workload_enhancement` in scenario YAML:

- **day_multiplier**: LogNormal(μ, σ) daily scalar on all arrival rates
- **autocorr_noise**: AR(1) process `z_t = ρ·z_{t-1} + ε_t` for temporal correlation
- **clustered_burst**: Markov-chain burst state machine on P3 arrivals
- **priority_mix_shift**: `redistribute` (shift P1 share up from P2+P3) or `amplify` (scale P1 up, total load increases)

All randomness uses a **deterministic RNG hierarchy** derived from a unified seed via `np.random.default_rng()` chain — zero global `np.random` calls.

### Reward Function

```
cost_t = elec_cost + qos_cost + switch_cost
elec_cost  = E_t · π_t           (energy × time-of-use price)
qos_cost   = Σ_k β_k·v_kt + ρ_k·overdue_kt
switch_cost = φ · |n_t - n_{t-1}| (ramping cost)

reward_t = -(w_e·elec/scale_e + w_q·qos/scale_q + w_s·switch/scale_s)
w = {1.0, 5.0, 0.2},  scale = {10.0, 10.0, 5.0}
```

### Key Parameters

| Parameter       | Value             | Description                       |
| --------------- | ----------------- | --------------------------------- |
| Server range    | [8, Nmax]         | scenario-dependent Nmax           |
| Action bins     | 21                | uniform mapping [Nmin, Nmax]      |
| Q-network       | [128, 128]        | 2-layer MLP, ReLU                 |
| Training eps    | 500               | ε: 1.0→0.05, decay=0.995          |
| ACI α_target    | 0.10              | target miscoverage rate           |
| ACI η           | 0.05              | interval width adaptation rate    |
| Conformal H     | 4                 | forecast/protect 4 steps (1 hour) |
| Multi-seed      | 3 train × 10 scen | mean±std, 95% CI via t-dist       |

---

## Results Output

After running experiments, the following CSV files are produced:

| File                       | Content                                            |
| -------------------------- | -------------------------------------------------- |
| `daily_results.csv`        | Per-day, per-method, per-seed metrics              |
| `experiment_summary.csv`   | Mean ± std ± 95% CI across seeds                   |
| `paired_tests.csv`         | Paired t-tests vs reference methods                |
| `aci_diagnostics.csv`      | Per-slot ACI coverage, alpha, interval width       |

---

## Project Structure

```
├── config.yaml                     # All system-default parameters
├── configs/scenarios/              # E0-E4 scenario YAML overlays
│   ├── E0_easy.yaml
│   ├── E1_normal_hard.yaml
│   ├── E2_distribution_shift.yaml
│   ├── E3_bursty_uncertainty.yaml
│   └── E4_capacity_cliff.yaml
├── README.md
├── PROJECT_ARCHITECTURE.txt        # Detailed architecture reference
├── main.py                         # Main experiment runner
├── generate_figures.py             # Publication-quality figures
├── analysis_plots.py               # Additional analysis plots
├── stress_test_scenarios.py        # S1/S2/S3 stress tests
├── _common.py                      # Shared: env builder, calibration
├── _heuristic_runner.py            # Shared heuristic evaluation loop
│
├── src/
│   ├── datacenter_env.py           # DataCenterEnv (Gym-like API)
│   ├── workload_generator.py       # Synthetic task generation + enhancements
│   ├── scenarios.py                # Scenario config loader & builder
│   ├── price_model.py              # TOU electricity price model
│   ├── data_preprocess.py          # Load/clean/split raw CSV data
│   ├── trace_loader.py             # Real-trace interface (Alibaba PAI)
│   ├── rl/
│   │   ├── dqn_agent.py            # QNetwork, ReplayBuffer, DQNAgent
│   │   ├── train_dqn.py            # Training/eval loops, EpisodeStats
│   │   └── augmenters.py           # Identity, Forecast, StaticConformal,
│   │                               #   Conformal augmenters
│   ├── conformal/
│   │   ├── aci.py                  # Adaptive Conformal Inference learner
│   │   ├── dtaci.py                # DtACI ensemble (appendix)
│   │   ├── forecaster.py           # RollingMean, Persistence, etc.
│   │   └── split_conformal.py      # Static split-conformal (baseline)
│   ├── safe_layer/
│   │   └── dtaci_action_shield.py  # Safety action shield (appendix)
│   ├── baselines/
│   │   ├── fixed_policy.py
│   │   ├── queue_greedy_policy.py
│   │   ├── price_aware_greedy_policy.py
│   │   ├── forecast_greedy_policy.py
│   │   └── conformal_greedy_policy.py
│   └── evaluation/
│       ├── metrics.py              # Aggregation, CI, paired tests
│       └── plot.py                 # Matplotlib helpers
│
└── tests/
    ├── _fixtures.py                # Tiny config for unit tests
    ├── test_env.py                  # Env invariants + strict action API
    ├── test_dtaci.py                # DtACI interval validity
    └── test_shield.py               # Shield invariants
```

## Running

```bash
pip install -r requirements.txt

# Main experiment (all 9 methods, multi-seed)
python main.py --scenario E1 --all

# Single method, quick smoke test
python main.py --scenario E1 --method aci_dqn --train-episodes 2 --test-days 5

# Figures
python generate_figures.py
python analysis_plots.py

# Stress tests
python stress_test_scenarios.py

# Unit tests
python -m pytest tests/ -v
```
