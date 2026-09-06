"""Policy configuration.

`load_config()` reads the policy from config.yml so modes and thresholds
switch without code edits, NeMo-style.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class PolicyConfig:
    """Weights and thresholds for the composite risk score."""

    # thresholds on composite risk -> RiskLevel
    low: float = 0.25
    medium: float = 0.50
    high: float = 0.75

    # weights for the signal-weighted sum
    w_reinforcement: float = 0.35
    w_sycophancy: float = 0.20
    w_delusion: float = 0.20
    w_conviction: float = 0.15
    w_isolation: float = 0.10

    # trajectory escalator: a rising delusion slope raises effective risk.
    # This term is what the single-turn-filter baseline lacks. Densities live
    # in [0,1], so a slope of ~0.1/turn is already a steep climb (full scale
    # in 10 turns) — the 3.0 gain turns that into a +0.3 risk boost.
    slope_boost: float = 3.0

    # The policy reads the slope over this many recent turns (None = whole
    # conversation). The reported DV always uses the whole-series formula;
    # this only affects how quickly the POLICY reacts to the current trend.
    slope_window: int | None = 4


@dataclass
class GuardConfig:
    """Everything `PsychosisGuard.from_config` needs: condition + policy."""

    condition: str = "combined"
    policy: PolicyConfig = field(default_factory=PolicyConfig)


def load_config(path: str | Path | None = None) -> GuardConfig:
    """Read a NeMo-style YAML config; None returns the defaults.

    Expected shape (all keys optional):

        condition: combined        # none | detect-only | single-turn-filter | A | B | combined
        policy:
          low: 0.25
          ...
          slope_boost: 3.0
    """
    if path is None:
        return GuardConfig()

    # configs must fail loudly on typos: a silently ignored key
    # (e.g. "conditon:") would run the wrong condition and corrupt results
    from .executor import VALID_MODES

    data = yaml.safe_load(Path(path).read_text()) or {}

    allowed_top = {"condition", "policy"}
    unknown_top = set(data) - allowed_top
    if unknown_top:
        raise ValueError(
            f"unknown top-level keys in {path}: {sorted(unknown_top)} "
            f"(allowed: {sorted(allowed_top)})"
        )

    condition = data.get("condition", "combined")
    if condition not in VALID_MODES:
        raise ValueError(
            f"unknown condition {condition!r} in {path}; "
            f"expected one of {sorted(VALID_MODES)}"
        )

    policy_data = data.get("policy") or {}
    known = {f.name for f in fields(PolicyConfig)}
    unknown = set(policy_data) - known
    if unknown:
        raise ValueError(f"unknown policy keys in {path}: {sorted(unknown)}")

    return GuardConfig(
        condition=condition,
        policy=PolicyConfig(**policy_data),
    )
