"""
src/price_model.py
------------------
Deterministic time-of-use electricity price curve. Generates the
``price`` series indexed by 15-min slot for one day.

The schedule is configurable in ``config.yaml`` under ``price``. Hours
not listed in ``high_hours`` or ``middle_hours`` default to the low
tier. The series length equals ``slots_per_day``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def build_daily_price(cfg: Dict) -> np.ndarray:
    """Return an array of length ``slots_per_day`` with CNY/kWh prices."""
    slots = cfg["time"]["slots_per_day"]
    low = cfg["price"]["low"]
    mid = cfg["price"]["middle"]
    high = cfg["price"]["high"]
    high_h = set(cfg["price"]["high_hours"])
    mid_h = set(cfg["price"]["middle_hours"])

    prices = np.full(slots, low, dtype=np.float64)
    for t in range(slots):
        hour = t // 4  # 4 slots per hour
        if hour in high_h:
            prices[t] = high
        elif hour in mid_h:
            prices[t] = mid
    return prices
