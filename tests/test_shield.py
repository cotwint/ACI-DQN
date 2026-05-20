"""
tests/test_shield.py
--------------------
Invariants of ``DtACIActionShield``:

* output is always in [Nmin, Nmax];
* when forecasts demand more capacity than the RL action provides,
  the shield raises the server count;
* when forecasts and queues are zero, the shield never *lowers* the
  RL action (it should be the identity).
"""

from __future__ import annotations

import numpy as np

from src.safe_layer.dtaci_action_shield import DtACIActionShield
from tests._fixtures import tiny_cfg


def test_output_always_within_bounds():
    cfg = tiny_cfg()
    shield = DtACIActionShield(cfg)
    Nmin, Nmax = cfg["server"]["Nmin"], cfg["server"]["Nmax"]
    K, H = cfg["qos"]["K"], cfg["conformal"]["horizon"]
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_rl = int(rng.integers(Nmin - 5, Nmax + 5))   # may be out of bounds
        q = rng.uniform(0, 50, size=K)
        # huge upper bounds -> would demand 10000 servers
        hi = rng.uniform(0, 1e3, size=(K, H))
        n_safe = shield.filter(n_rl, q, hi)
        assert Nmin <= n_safe <= Nmax


def test_shield_raises_when_demand_high():
    cfg = tiny_cfg()
    shield = DtACIActionShield(cfg)
    K, H = cfg["qos"]["K"], cfg["conformal"]["horizon"]
    q = np.zeros(K)
    # Lambda upper at clip limit (20.0) for P1+P2 across horizon -> high demand.
    hi = np.zeros((K, H))
    hi[0, :] = 20.0
    hi[1, :] = 20.0
    n_rl = cfg["server"]["Nmin"]
    n_safe = shield.filter(n_rl, q, hi)
    # W_U = 4*20*0.60 + 4*20*1.20 = 48+96 = 144, n_req = ceil(144/4) = 36
    assert n_safe > n_rl, f"shield should raise when demand is high, got {n_safe}"
    assert n_safe == 36, f"expected 36 under clip-limited demand, got {n_safe}"


def test_shield_identity_when_no_demand():
    cfg = tiny_cfg()
    shield = DtACIActionShield(cfg)
    K, H = cfg["qos"]["K"], cfg["conformal"]["horizon"]
    q = np.zeros(K)
    hi = np.zeros((K, H))
    n_rl = (cfg["server"]["Nmin"] + cfg["server"]["Nmax"]) // 2
    n_safe = shield.filter(n_rl, q, hi)
    assert n_safe == n_rl


def test_shield_logs_modifications():
    cfg = tiny_cfg()
    shield = DtACIActionShield(cfg)
    K, H = cfg["qos"]["K"], cfg["conformal"]["horizon"]
    hi = np.zeros((K, H))
    # 5 no-mod steps
    for _ in range(5):
        shield.filter(20, np.zeros(K), hi)
    # 3 modifying steps
    hi[0, :] = 1e6
    for _ in range(3):
        shield.filter(8, np.zeros(K), hi)
    assert shield.log.n_steps == 8
    assert shield.log.n_mods == 3
    assert 0.0 < shield.log.mod_rate <= 1.0
