"""Unit tests for InterventionPolicyEngine (composite risk + level decision)."""

import pytest

from psychosis_guard.config import PolicyConfig
from psychosis_guard.policy import InterventionPolicyEngine
from psychosis_guard.types import RiskLevel, Signals


def _engine(**overrides) -> InterventionPolicyEngine:
    return InterventionPolicyEngine(PolicyConfig(**overrides))


class TestCompositeRisk:
    def test_zero_signals_zero_risk(self):
        assert _engine().composite_risk(Signals(), slope=0.0) == 0.0

    def test_weighted_sum_matches_config(self):
        cfg = PolicyConfig()
        signals = Signals(reinforcement=1.0, sycophancy=0.5, delusion_density=0.5)
        expected = cfg.w_reinforcement * 1.0 + cfg.w_sycophancy * 0.5 + cfg.w_delusion * 0.5
        assert _engine().composite_risk(signals, 0.0) == pytest.approx(expected)

    def test_positive_slope_boosts_risk(self):
        signals = Signals(delusion_density=0.5)
        flat = _engine().composite_risk(signals, slope=0.0)
        rising = _engine().composite_risk(signals, slope=0.2)
        assert rising == pytest.approx(flat + PolicyConfig().slope_boost * 0.2)

    def test_negative_slope_is_not_rewarded(self):
        signals = Signals(delusion_density=0.5)
        assert _engine().composite_risk(signals, -0.4) == _engine().composite_risk(signals, 0.0)

    def test_risk_capped_at_one(self):
        maxed = Signals(reinforcement=1, sycophancy=1, delusion_density=1,
                        conviction=1, isolation=1)
        assert _engine().composite_risk(maxed, slope=1.0) == 1.0


class TestDecide:
    @pytest.mark.parametrize(
        ("risk", "level"),
        [
            (0.0, RiskLevel.NONE),
            (0.24, RiskLevel.NONE),
            (0.25, RiskLevel.LOW),   # thresholds are inclusive
            (0.49, RiskLevel.LOW),
            (0.50, RiskLevel.MEDIUM),
            (0.74, RiskLevel.MEDIUM),
            (0.75, RiskLevel.HIGH),
            (1.0, RiskLevel.HIGH),
        ],
    )
    def test_threshold_boundaries(self, risk, level):
        assert _engine().decide(risk) == level

    def test_custom_thresholds_respected(self):
        engine = _engine(low=0.1, medium=0.2, high=0.3)
        assert engine.decide(0.15) == RiskLevel.LOW
        assert engine.decide(0.35) == RiskLevel.HIGH
