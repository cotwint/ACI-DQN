"""
src/conformal/forecaster.py
---------------------------
Lightweight time-series forecasters used by the conformal layer.

Targets that we may forecast:
* Task arrival rates lambda_k(t) for k = 1,2,3
* Electricity price pi(t)
* Normalised regional load x(t)

We deliberately keep these *simple* so that the project runs without
sklearn. If sklearn is installed and ``forecaster='sklearn_ridge'`` is
selected in config, ``SklearnRidgeForecaster`` will be used; otherwise
falls back to rolling mean.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Persistence: y_hat(t+h) = y(t)
# ---------------------------------------------------------------------------

class PersistenceForecaster:
    name = "persistence"

    def __init__(self, horizon: int = 1):
        self.horizon = int(horizon)

    def predict(self, history: np.ndarray, h: int = 1) -> float:
        if history.size == 0:
            return 0.0
        return float(history[-1])


# ---------------------------------------------------------------------------
# Rolling mean
# ---------------------------------------------------------------------------

class RollingMeanForecaster:
    name = "rolling_mean"

    def __init__(self, window: int = 8, horizon: int = 1):
        self.window = int(window)
        self.horizon = int(horizon)

    def predict(self, history: np.ndarray, h: int = 1) -> float:
        if history.size == 0:
            return 0.0
        w = min(self.window, history.size)
        return float(np.mean(history[-w:]))


# ---------------------------------------------------------------------------
# Seasonal naive: y_hat(t+h) = y(t+h-T) (one-day lag)
# ---------------------------------------------------------------------------

class SeasonalNaiveForecaster:
    name = "seasonal_naive"

    def __init__(self, period: int = 96, horizon: int = 1):
        self.period = int(period)
        self.horizon = int(horizon)

    def predict(self, history: np.ndarray, h: int = 1) -> float:
        if history.size == 0:
            return 0.0
        idx = -self.period + (h - 1)
        if -idx > history.size:
            return float(history[-1])
        return float(history[idx])


# ---------------------------------------------------------------------------
# Optional sklearn ridge: only used if sklearn is installed.
# ---------------------------------------------------------------------------

class SklearnRidgeForecaster:
    """Ridge regression on the last ``window`` lags. Lazy-trained."""
    name = "sklearn_ridge"

    def __init__(self, window: int = 16, horizon: int = 1, alpha: float = 1.0):
        try:
            from sklearn.linear_model import Ridge   # noqa: F401
        except ImportError as e:
            raise ImportError("sklearn is required for SklearnRidgeForecaster") from e
        self.window = int(window)
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self._model = None

    def _fit(self, history: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        if history.size < self.window + 1:
            self._model = None
            return
        X = np.stack([history[i:i + self.window]
                      for i in range(history.size - self.window)])
        y = history[self.window:]
        self._model = Ridge(alpha=self.alpha).fit(X, y)

    def predict(self, history: np.ndarray, h: int = 1) -> float:
        if history.size < self.window + 1:
            return float(history[-1]) if history.size else 0.0
        self._fit(history)
        if self._model is None:
            return float(history[-1])
        return float(self._model.predict(history[-self.window:][None, :])[0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_forecaster(kind: str, **kwargs):
    if kind == "persistence":
        return PersistenceForecaster(**kwargs)
    if kind == "rolling_mean":
        return RollingMeanForecaster(**kwargs)
    if kind == "seasonal_naive":
        return SeasonalNaiveForecaster(**kwargs)
    if kind == "sklearn_ridge":
        return SklearnRidgeForecaster(**kwargs)
    raise ValueError(f"Unknown forecaster {kind}")
