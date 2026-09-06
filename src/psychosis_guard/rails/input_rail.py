"""Stage 1 — INPUT RAIL (NeMo analogue: input rail).

Pre-response risk estimate from the user turn alone, before the chatbot is
called. Uses only the API-free lexical features plus the PRIOR trajectory
slope, so Mode A can pick a system prompt without an extra LLM round-trip.
"""
from __future__ import annotations

from .. import lexicons as lex
from ..policy import InterventionPolicyEngine
from ..types import RiskLevel, Signals, TurnRecord


class InputRail:
    STAGE = "stage1:input"

    def __init__(self, policy: InterventionPolicyEngine) -> None:
        self.policy = policy

    def run(self, record: TurnRecord, prior_slope: float) -> RiskLevel:
        """Estimate pre-response risk; returns the pre-level for Stage 2."""
        pre_signals = Signals(
            delusion_density=lex.lexical_density(record.user_text, lex.DELUSION),
            conviction=lex.conviction_score(record.user_text),
        )
        pre_risk = self.policy.composite_risk(pre_signals, prior_slope)
        pre_level = self.policy.decide(pre_risk)
        record.trace(
            self.STAGE,
            f"pre_risk={pre_risk:.3f} pre_level={pre_level.name} "
            f"(prior_slope={prior_slope:.3f})",
        )
        return pre_level
