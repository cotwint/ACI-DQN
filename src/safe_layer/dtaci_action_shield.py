"""
src/safe_layer/dtaci_action_shield.py
-------------------------------------
Conservative action-shielding wrapper that applies *after* the RL policy
proposes ``n^{RL}_t``. The shield uses DtACI upper bounds for the next
H steps of high-priority task arrival rates to lower-bound the number
of servers required:

    W^U(t) = sum_{k in K'} [ Q_k(t) + sum_{h=1}^H lambda_k^U(t+h) * c_bar_k ]

    n^safe_t = max( n^RL_t, ceil( W^U(t) / (mu * H * dt) ) )

clamped to [Nmin, Nmax]. ``K'`` defaults to {1,2} (protect P1+P2); P3
tolerates more queuing so we do not over-provision for it.

We log every modification so the user can plot ``shield_mod_rate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ShieldLog:
    n_steps: int = 0
    n_mods: int = 0
    mod_sizes: List[float] = field(default_factory=list)

    @property
    def mod_rate(self) -> float:
        return self.n_mods / max(1, self.n_steps)

    @property
    def avg_mod_size(self) -> float:
        return float(np.mean(self.mod_sizes)) if self.mod_sizes else 0.0


class DtACIActionShield:
    """Conservative safety filter on top of a discrete RL policy.

    Parameters
    ----------
    cfg : full config dict
    protect_priorities : optional override of config (1-based indices).
    horizon : H (default from cfg.conformal.horizon).
    """

    def __init__(self, cfg: Dict,
                 protect_priorities: list | None = None,
                 horizon: int | None = None):
        self.cfg = cfg
        self.protect = (protect_priorities
                        if protect_priorities is not None
                        else cfg["conformal"]["protect_priorities"])
        # Convert 1-based to 0-based.
        self.protect0 = [int(p) - 1 for p in self.protect]
        self.H = int(horizon if horizon is not None
                     else cfg["conformal"]["horizon"])
        self.cap = float(cfg["server"]["cap_per_server"])
        self.Nmin = int(cfg["server"]["Nmin"])
        self.Nmax = int(cfg["server"]["Nmax"])
        self.service_mean = np.asarray(cfg["qos"]["service_mean"],
                                       dtype=np.float64)
        # Per-server capacity over H slots in compute units.
        # cap_per_server is compute units per server per SLOT, so over
        # H slots one server provides cap * H compute units.
        self.denom = max(1e-9, self.cap * self.H)
        self.upper_clip_method = cfg["conformal"].get("upper_bound_clip_method", "p95")
        self.lambda_scale = float(cfg["conformal"].get("lambda_scale", 20.0))
        self.log = ShieldLog()

    # ------------------------------------------------------------------
    def filter(self,
               n_rl: int,
               queue_len: np.ndarray,
               lambda_upper: np.ndarray) -> int:
        """Return n_safe and update shield log.

        Parameters
        ----------
        n_rl : the RL policy's proposed number of active servers
        queue_len : (K,) current queue lengths Q_k(t)
        lambda_upper : (K, H) per-priority upper-bound forecasts of
                       arrival rates for the next H steps
        """
        # Total upper-bound work that we must guarantee for protected
        # priorities. Note: ``queue_len`` counts tasks, so we multiply
        # by c_bar_k to convert to compute units. Backlog work would be
        # more accurate, but the conformal interval covers arrivals
        # only -- queues are *current* state, treated deterministically.
        c_bar = self.service_mean   # average compute per task per priority
        # Clip extreme lambda_upper to prevent over-provisioning.
        clip_val = self.lambda_scale if self.upper_clip_method == "p95" else None
        W_U = 0.0
        for k0 in self.protect0:
            lam_u_k = lambda_upper[k0].copy()
            if clip_val is not None:
                lam_u_k = np.clip(lam_u_k, 0.0, clip_val)
            W_U += queue_len[k0] * c_bar[k0]
            W_U += float(np.sum(lam_u_k)) * c_bar[k0]

        n_required = int(np.ceil(W_U / self.denom))
        n_safe = max(int(n_rl), n_required)
        n_safe = int(np.clip(n_safe, self.Nmin, self.Nmax))

        self.log.n_steps += 1
        if n_safe != int(n_rl):
            self.log.n_mods += 1
            self.log.mod_sizes.append(float(n_safe - int(n_rl)))
        return n_safe

    # ------------------------------------------------------------------
    def reset_log(self) -> None:
        self.log = ShieldLog()
