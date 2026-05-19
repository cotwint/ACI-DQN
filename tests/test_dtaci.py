"""
tests/test_dtaci.py
-------------------
Validate DtACI's basic contract:

* warm_up + interval emits a numeric (lo, hi) pair with lo <= hi
* repeated update() pushes the empirical coverage toward 1 - alpha
* expert mixture weights remain a probability distribution
"""

from __future__ import annotations

import numpy as np

from src.conformal.dtaci import DtACI


def test_interval_well_formed_after_warmup():
    rng = np.random.default_rng(0)
    d = DtACI(alpha_target=0.10)
    y_hat = rng.normal(size=200)
    y = y_hat + rng.normal(scale=1.0, size=200)
    d.warm_up(y_hat, y)
    lo, hi = d.interval(0.0)
    assert lo <= hi
    assert np.isfinite(lo) and np.isfinite(hi)


def test_online_coverage_close_to_target():
    rng = np.random.default_rng(1)
    d = DtACI(alpha_target=0.10)
    # warm up
    y_hat_w = rng.normal(size=100)
    y_w = y_hat_w + rng.normal(scale=1.0, size=100)
    d.warm_up(y_hat_w, y_w)

    # Stream a moderate number of samples through the online loop
    N = 1500
    y_hats = rng.normal(size=N)
    y_true = y_hats + rng.normal(scale=1.0, size=N)
    for yh, yt in zip(y_hats, y_true):
        lo, hi = d.interval(float(yh))
        d.update(float(yh), float(yt), lo, hi)

    cov = np.mean(d.coverage_log)
    # Loose check: coverage roughly within +/-5pp of 90%
    assert 0.83 <= cov <= 0.97, f"coverage {cov} not near 0.90"


def test_expert_weights_are_a_distribution():
    d = DtACI()
    w = d.weights
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    # After some updates the weights should still sum to 1
    rng = np.random.default_rng(2)
    y_hat_w = rng.normal(size=50)
    y_w = y_hat_w + rng.normal(scale=1.0, size=50)
    d.warm_up(y_hat_w, y_w)
    for _ in range(100):
        yh = float(rng.normal())
        yt = yh + float(rng.normal(scale=1.0))
        lo, hi = d.interval(yh)
        d.update(yh, yt, lo, hi)
    assert np.isclose(d.weights.sum(), 1.0, atol=1e-6)
