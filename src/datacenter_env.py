"""
src/datacenter_env.py
---------------------
Discrete-time data center environment with 3 priority queues, preemptive
service, deadline-aware QoS cost and time-of-use electricity price.

The dynamics mirror the existing Octave reference (``simulate_datacenter.m``,
``dispatch_priority.m``) but are rewritten cleanly in NumPy/Python so that
DQN / ACI-DQN / DtACI-DQN can call ``step`` without a MATLAB runtime.

State (length 11):
    [Q1, Q2, Q3, B1, B2, B3, price, x_load, n_prev,
     sin(2*pi*t/T), cos(2*pi*t/T)]

Action: integer in {0..action_bins-1}, mapped linearly to Nmin..Nmax.

Reward: r_t = -(C_elec + C_QoS + C_sw)  (scaled later by the RL trainer).

The env does not depend on gymnasium so it is easy to use from plain
loops or vector envs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .price_model import build_daily_price
from .workload_generator import (
    DayWorkload, Task, generate_day, generate_day_enhanced,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def action_to_n(action: int, cfg: Dict) -> int:
    """Map discrete action index to number of active servers.

    No silent clipping — caller must ensure *action* is in [0, bins-1].
    """
    bins = cfg["rl"]["action_bins"]
    Nmin = cfg["server"]["Nmin"]
    Nmax = cfg["server"]["Nmax"]
    n = Nmin + (Nmax - Nmin) * action / (bins - 1)
    return int(round(n))


def n_to_action(n: int, cfg: Dict) -> int:
    """Inverse of ``action_to_n`` -- nearest action index for a given n."""
    bins = cfg["rl"]["action_bins"]
    Nmin = cfg["server"]["Nmin"]
    Nmax = cfg["server"]["Nmax"]
    n = float(np.clip(n, Nmin, Nmax))
    a = (n - Nmin) / max(Nmax - Nmin, 1) * (bins - 1)
    return int(round(a))


# ---------------------------------------------------------------------------
# Step info container
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    t: int = 0
    n_active: int = 0
    util: float = 0.0
    it_power_kw: float = 0.0
    facility_power_kw: float = 0.0
    energy_kwh: float = 0.0
    elec_cost: float = 0.0
    qos_cost: float = 0.0
    switch_cost: float = 0.0
    total_cost: float = 0.0
    arrivals: np.ndarray = field(default_factory=lambda: np.zeros(3))
    served_work: np.ndarray = field(default_factory=lambda: np.zeros(3))
    completed: np.ndarray = field(default_factory=lambda: np.zeros(3))
    violations: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delay_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    overdue_pending: np.ndarray = field(default_factory=lambda: np.zeros(3))
    qos_cost_per_priority: np.ndarray = field(default_factory=lambda: np.zeros(3))
    queue_len: np.ndarray = field(default_factory=lambda: np.zeros(3))
    backlog_work: np.ndarray = field(default_factory=lambda: np.zeros(3))
    arrival_work: np.ndarray = field(default_factory=lambda: np.zeros(3))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DataCenterEnv:
    """
    OpenAI Gym-like (but standalone) data center environment.

    Parameters
    ----------
    cfg : full config dict from ``utils.load_config``
    day_load_matrix : (D, T) regional load curve, kW. Used to derive x(t).
    day_norm_matrix : (D, T) per-day min-max normalised load in [0,1].
    base_seed : int for workload generation seeds (each day uses base+day_idx).
    external_workloads : optional dict mapping day_index -> DayWorkload.
        When provided, ``reset()`` uses the pre-built workload instead of
        calling ``generate_day()``. This is the interface for real trace data.
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 cfg: Dict,
                 day_load_matrix: np.ndarray,
                 day_norm_matrix: np.ndarray,
                 base_seed: int = 0,
                 external_workloads: Optional[Dict[int, DayWorkload]] = None):
        self.cfg = cfg
        self.day_load_matrix = day_load_matrix
        self.day_norm_matrix = day_norm_matrix
        self.base_seed = int(base_seed)
        self._external_workloads = external_workloads

        self.T = int(cfg["time"]["slots_per_day"])
        self.dt = float(cfg["time"]["dt_hour"])
        self.K = int(cfg["qos"]["K"])
        self.Nmin = int(cfg["server"]["Nmin"])
        self.Nmax = int(cfg["server"]["Nmax"])
        self.cap = float(cfg["server"]["cap_per_server"])
        self.P_idle = float(cfg["power"]["P_idle"])
        self.P_peak = float(cfg["power"]["P_peak"])
        self.P_fixed = float(cfg["power"]["P_fixed"])
        self.PUE = float(cfg["power"]["PUE"])
        self.gamma = float(cfg["power"]["power_gamma"])
        self.c_sw = float(cfg["power"]["switch_cost"])
        self.beta = np.asarray(cfg["qos"]["sla_penalty"], dtype=np.float64)
        self.rho = np.asarray(cfg["qos"]["overdue_penalty"], dtype=np.float64)
        self.deadlines = np.asarray(cfg["qos"]["deadline_slots"], dtype=int)

        # Reward component weights and scales.
        rw = cfg["rl"].get("reward_weights", {})
        self.w_elec = float(rw.get("elec", 1.0))
        self.w_qos = float(rw.get("qos", 5.0))
        self.w_switch = float(rw.get("switch", 0.2))
        rs = cfg["rl"].get("reward_scales", {})
        self.elec_scale = float(rs.get("elec", 10.0))
        self.qos_scale = float(rs.get("qos", 10.0))
        self.switch_scale = float(rs.get("switch", 5.0))

        self.price_curve = build_daily_price(cfg)
        self._workload_params: Dict | None = None  # set via set_workload_params()
        # State: Q1,Q2,Q3, B1,B2,B3, near_deadline1,2,3, min_slack1,2,3,
        #        price, x, n_prev, sin, cos = 17 dims
        self.observation_dim = 3 + 3 + 3 + 3 + 1 + 1 + 1 + 2
        self.action_dim = int(cfg["rl"]["action_bins"])

        # State that mutates each step.
        self.day_index: Optional[int] = None
        self.workload: Optional[DayWorkload] = None
        self.x_curve: Optional[np.ndarray] = None
        self.queues: List[List[Task]] = [[] for _ in range(self.K)]
        self.t: int = 0
        self.n_prev: int = self.Nmin
        self.done: bool = False
        self.history: List[StepInfo] = []

    # -----------------------------------------------------------------
    # Public mutation API (only way to change config-derived fields)
    # -----------------------------------------------------------------

    def set_price_curve(self, curve: np.ndarray) -> None:
        """Replace the daily price curve.  *curve* shape must be ``(T,)``."""
        if curve.shape != (self.T,):
            raise ValueError(f"price_curve shape {curve.shape}, expected ({self.T},)")
        self.price_curve = curve.astype(np.float64).copy()

    def set_workload_params(self, params: Dict) -> None:
        """Store workload enhancement params used by ``reset()``.

        When set, ``reset()`` calls ``generate_day_enhanced()`` instead of
        the plain ``generate_day()``.
        """
        self._workload_params = params

    # -----------------------------------------------------------------
    # Public Gym-style API
    # -----------------------------------------------------------------

    def reset(self, day_index: int,
              seed: Optional[int] = None,
              workload_override: Optional[DayWorkload] = None
              ) -> Tuple[np.ndarray, Dict]:
        """Reset env to start of one day.

        Workload resolution order:
        1. ``workload_override`` (one-shot, e.g. stress test injection)
        2. ``self._external_workloads[day_index]`` (real-trace interface)
        3. ``generate_day()`` (synthetic, default)
        """
        if not (0 <= day_index < self.day_load_matrix.shape[0]):
            raise IndexError(f"day_index {day_index} out of range")
        self.day_index = day_index
        self.x_curve = self.day_norm_matrix[day_index].copy()

        if workload_override is not None:
            self.workload = workload_override
        elif (self._external_workloads is not None
              and day_index in self._external_workloads):
            self.workload = self._external_workloads[day_index]
        else:
            wseed = self.base_seed + day_index if seed is None else int(seed)
            if self._workload_params:
                self.workload = generate_day_enhanced(
                    self.x_curve, self.cfg, wseed, self._workload_params,
                )
            else:
                self.workload = generate_day(self.x_curve, self.cfg, wseed)

        self.queues = [[] for _ in range(self.K)]
        self.t = 0
        self.n_prev = self.Nmin
        self.done = False
        self.history = []

        return self.get_state(), {"day_index": day_index}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance one 15-minute slot.

        *action* must be an action index in ``[0, action_bins-1]``.
        Passing a server count (e.g. 68) will raise ``ValueError``.
        """
        if self.done:
            raise RuntimeError("step() called after done. Call reset().")
        if self.workload is None or self.x_curve is None:
            raise RuntimeError("Env not reset().")

        bins = self.action_dim
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= bins:
            raise ValueError(
                f"env.step() expects action index in [0, {bins - 1}], "
                f"got {action}. Use n_to_action() to convert server count "
                f"to action index before calling env.step()."
            )

        t = self.t
        n_active = action_to_n(action, self.cfg)
        n_active = self._apply_ramp_limit(n_active)

        # ---- task arrivals at slot t ----
        info = StepInfo(t=t, n_active=n_active)
        for k in range(self.K):
            new_tasks = self.workload.tasks_per_slot[k][t]
            self.queues[k].extend(new_tasks)
            info.arrivals[k] = len(new_tasks)
            info.arrival_work[k] = self.workload.arrival_work[k, t]

        # ---- dispatch: strict priority + EDF within priority ----
        served_work, completed, violations, delay_sum = self._dispatch(
            n_active, t
        )
        info.served_work = served_work
        info.completed = completed
        info.violations = violations
        info.delay_sum = delay_sum

        # ---- capacity / utilisation / power ----
        cap = n_active * self.cap
        total_served = float(served_work.sum())
        util = min(1.0, total_served / cap) if cap > 0 else 0.0
        srv_power = self.P_idle + (self.P_peak - self.P_idle) * (util ** self.gamma)
        it_power = n_active * srv_power + self.P_fixed
        facility_power = self.PUE * it_power
        energy = facility_power * self.dt
        elec_cost = energy * self.price_curve[t]

        # ---- QoS cost ----
        overdue_pending = self._count_overdue_pending(t)
        qos_per_priority = self.beta * violations + self.rho * overdue_pending
        qos_cost = float(qos_per_priority.sum())

        # ---- switching cost ----
        switch_cost = self.c_sw * abs(n_active - self.n_prev)

        total_cost = elec_cost + qos_cost + switch_cost
        weighted_cost = (
            self.w_elec * elec_cost / self.elec_scale
            + self.w_qos * qos_cost / self.qos_scale
            + self.w_switch * switch_cost / self.switch_scale
        )
        reward = -weighted_cost

        # ---- record per-step diagnostics ----
        info.util = util
        info.it_power_kw = it_power
        info.facility_power_kw = facility_power
        info.energy_kwh = energy
        info.elec_cost = elec_cost
        info.qos_cost = qos_cost
        info.switch_cost = switch_cost
        info.total_cost = total_cost
        info.overdue_pending = overdue_pending
        info.qos_cost_per_priority = qos_per_priority
        info.queue_len = np.array([len(q) for q in self.queues], dtype=np.float64)
        info.backlog_work = np.array(
            [sum(task.remaining for task in q) for q in self.queues],
            dtype=np.float64,
        )
        self.history.append(info)
        self.n_prev = n_active
        self.t += 1
        if self.t >= self.T:
            self.done = True

        return self.get_state(), float(reward), bool(self.done), info.__dict__

    # -----------------------------------------------------------------
    # State construction
    # -----------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Build state vector with QoS-aware features and normalisation."""
        K = self.K
        Q = np.array([len(q) for q in self.queues], dtype=np.float64)
        B = np.array([sum(task.remaining for task in q) for q in self.queues],
                     dtype=np.float64)
        t = min(self.t, self.T - 1)

        # Near-deadline tasks and minimum slack per priority.
        near_deadline = np.zeros(K, dtype=np.float64)
        min_slack = np.full(K, np.nan, dtype=np.float64)
        for k in range(K):
            if self.queues[k]:
                slacks = np.array([task.deadline - t + 1 for task in self.queues[k]])
                min_slack[k] = float(slacks.min())
                near_thresh = max(1, int(np.ceil(0.25 * self.deadlines[k])))
                near_deadline[k] = float(np.sum(slacks <= near_thresh))
            else:
                min_slack[k] = float(self.deadlines[k])

        price = self.price_curve[t]
        x = float(self.x_curve[t]) if self.x_curve is not None else 0.0
        ang = 2.0 * np.pi * t / self.T

        # Normalisation divisors for numerical stability.
        q_norm = 500.0
        b_norm = 500.0
        near_norm = np.maximum(self.deadlines.astype(np.float64), 1.0)
        slack_norm = np.maximum(self.deadlines.astype(np.float64), 1.0)
        price_norm = 1.0
        n_norm = float(self.Nmax)

        s = np.concatenate([
            Q / q_norm,
            B / b_norm,
            near_deadline / near_norm,
            np.nan_to_num(min_slack / slack_norm, nan=1.0),
            np.array([price / price_norm, x, self.n_prev / n_norm,
                      np.sin(ang), np.cos(ang)], dtype=np.float64),
        ])
        return s.astype(np.float32)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _apply_ramp_limit(self, n: int) -> int:
        lim = int(self.cfg["server"]["ramp_limit"])
        low = self.n_prev - lim
        high = self.n_prev + lim
        n = max(low, min(high, n))
        n = max(self.Nmin, min(self.Nmax, n))
        return n

    def _dispatch(self, n_active: int, t: int):
        """Strict-priority + EDF preemptive dispatch.

        Returns served_work, completed, violations, delay_sum (per priority).
        """
        cap = n_active * self.cap
        served_work = np.zeros(self.K)
        completed = np.zeros(self.K)
        violations = np.zeros(self.K)
        delay_sum = np.zeros(self.K)

        for k in range(self.K):
            if cap <= 1e-12 or not self.queues[k]:
                continue
            # EDF: stable sort by deadline ascending
            self.queues[k].sort(key=lambda task: task.deadline)
            done_idx = []
            for i, task in enumerate(self.queues[k]):
                if cap <= 1e-12:
                    break
                served = min(cap, task.remaining)
                task.remaining -= served
                cap -= served
                served_work[k] += served
                if task.remaining <= 1e-9:
                    delay_slots = t - task.arrival + 1
                    violated = float(t > task.deadline)
                    completed[k] += 1
                    violations[k] += violated
                    delay_sum[k] += delay_slots
                    done_idx.append(i)
            # Remove completed tasks in reverse order to keep indices valid.
            for i in reversed(done_idx):
                self.queues[k].pop(i)
        return served_work, completed, violations, delay_sum

    def _count_overdue_pending(self, t: int) -> np.ndarray:
        out = np.zeros(self.K)
        for k in range(self.K):
            out[k] = sum(1 for task in self.queues[k] if t > task.deadline)
        return out

    # -----------------------------------------------------------------
    # Convenience accessors for analysis / plotting
    # -----------------------------------------------------------------

    def history_as_arrays(self) -> Dict[str, np.ndarray]:
        """Stack StepInfo records into numpy arrays for easy aggregation."""
        if not self.history:
            return {}
        keys_scalar = ["t", "n_active", "util", "it_power_kw",
                       "facility_power_kw", "energy_kwh",
                       "elec_cost", "qos_cost", "switch_cost", "total_cost"]
        keys_vec = ["arrivals", "served_work", "completed", "violations",
                    "delay_sum", "overdue_pending", "queue_len",
                    "backlog_work", "arrival_work", "qos_cost_per_priority"]
        out = {k: np.array([getattr(s, k) for s in self.history])
               for k in keys_scalar}
        for k in keys_vec:
            out[k] = np.stack([getattr(s, k) for s in self.history], axis=0)
        return out
