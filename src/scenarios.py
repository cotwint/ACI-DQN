"""
src/scenarios.py
----------------
Scenario builder for E0-E4 benchmark framework.

Loads scenario YAML overlays, deep-merges with base config, and resolves
phase-specific overrides (train / calibration / test) declared under the
``phases:`` key in each scenario file.

Usage::

    from src.scenarios import build_scenario_config, get_phase_config

    scenario_cfg = build_scenario_config("E1", base_cfg)
    train_cfg = get_phase_config(scenario_cfg, "train")
    test_cfg  = get_phase_config(scenario_cfg, "test")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Recursively merge *overlay* into a deep copy of *base*.

    Dicts are merged recursively; all other values are overwritten.
    """
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _scenario_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "scenarios"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_scenario_overlay(scenario_id: str) -> Dict[str, Any]:
    """Load a scenario YAML file.

    Returns the raw overlay dict (no base config merging).
    """
    path = _scenario_dir() / f"{scenario_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario config not found: {path}\n"
            f"Available scenarios: {[p.stem for p in _scenario_dir().glob('*.yaml')]}"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_scenario_config(scenario_id: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy *base_cfg* and deep-merge the scenario overlay.

    Returns a new, independent config dict. The original *base_cfg* is never
    mutated. Phase overrides remain nested under ``phases.*`` — call
    ``get_phase_config()`` to resolve them.
    """
    overlay = load_scenario_overlay(scenario_id)
    # Remove the phases key before merging to avoid polluting top-level keys.
    phase_data = overlay.pop("phases", {})
    merged = _deep_merge(base_cfg, overlay)
    # Restore phases as a nested key.
    merged["_phases"] = phase_data
    return merged


def get_phase_config(scenario_cfg: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """Return the effective config for *phase*.

    Merges the scenario root with ``_phases.{phase}`` overrides.
    *phase* must be one of ``"train"``, ``"calibration"``, ``"test"``.
    """
    if phase not in ("train", "calibration", "test"):
        raise ValueError(f"Unknown phase '{phase}'; expected train/calibration/test")
    phase_overrides = scenario_cfg.get("_phases", {}).get(phase, {})
    if not phase_overrides:
        return copy.deepcopy(scenario_cfg)
    return _deep_merge(scenario_cfg, phase_overrides)


def apply_scenario_to_env(env, phase_cfg: Dict[str, Any]) -> None:
    """Configure *env* for a specific scenario phase.

    Calls only public API methods: ``set_price_curve()`` and
    ``set_workload_params()``.
    """
    # Price curve: keep env default for now (scenarios can override later).
    if "price_override" in phase_cfg:
        import numpy as np
        env.set_price_curve(np.asarray(phase_cfg["price_override"], dtype=np.float64))

    # Workload enhancement params
    wl_enh = phase_cfg.get("workload_enhancement", {})
    env.set_workload_params(wl_enh)


def list_available_scenarios() -> list:
    """Return sorted list of available scenario IDs."""
    return sorted(p.stem for p in _scenario_dir().glob("*.yaml"))
