"""(3) InterventionPolicyEngine — graduated response keyed to risk.

Turns the per-turn Signals plus the trajectory slope into a composite risk
score, then maps that score onto a RiskLevel. The slope term is the
trajectory-aware part: a conversation that is *escalating* is treated as
riskier than the same signals in a flat conversation.
"""
from __future__ import annotations

from .config import PolicyConfig
from .types import RiskLevel, Signals


class InterventionPolicyEngine:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.cfg = config or PolicyConfig()

    def composite_risk(self, signals: Signals, slope: float) -> float:
        """Weighted sum of signals + slope boost, capped to [0, 1]."""
        c = self.cfg
        base = (
            c.w_reinforcement * signals.reinforcement
            + c.w_sycophancy * signals.sycophancy
            + c.w_delusion * signals.delusion_density
            + c.w_conviction * signals.conviction
            + c.w_isolation * signals.isolation
        )
        # only an ESCALATING trajectory raises risk; a falling slope is not
        # rewarded, so interventions do not switch off while density is high
        base += c.slope_boost * max(slope, 0.0)
        return min(base, 1.0)

    def decide(self, risk: float) -> RiskLevel:
        c = self.cfg
        if risk >= c.high:
            return RiskLevel.HIGH
        if risk >= c.medium:
            return RiskLevel.MEDIUM
        if risk >= c.low:
            return RiskLevel.LOW
        return RiskLevel.NONE
