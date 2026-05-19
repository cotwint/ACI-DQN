"""Constant-capacity baseline: always keep the same number of servers on."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..datacenter_env import n_to_action


class FixedPolicy:
    name = "fixed"

    def __init__(self, cfg: Dict, n_fixed: Optional[int] = None):
        Nmin = int(cfg["server"]["Nmin"])
        Nmax = int(cfg["server"]["Nmax"])
        if n_fixed is None:
            n_fixed = (Nmin + Nmax) // 2
        self.n_fixed = int(np.clip(n_fixed, Nmin, Nmax))
        self.cfg = cfg

    def act(self, state: np.ndarray, env) -> int:
        return n_to_action(self.n_fixed, self.cfg)

    def reset(self):
        pass
