"""Stage 5 — ACTION RAIL (NeMo analogue: actions).

Computes the final composite risk (signals + trajectory slope), decides the
intervention level, and applies the Mode-B post-processing.
"""
from __future__ import annotations

from ..executor import InterventionExecutor
from ..policy import InterventionPolicyEngine
from ..types import RiskLevel, Turn, TurnRecord


class ActionRail:
    STAGE = "stage5:action"

    def __init__(
        self, policy: InterventionPolicyEngine, executor: InterventionExecutor
    ) -> None:
        self.policy = policy
        self.executor = executor

    def run(self, history: list[Turn], record: TurnRecord, slope: float) -> str:
        risk = self.policy.composite_risk(record.signals, slope)
        level = self.policy.decide(risk)
        record.composite_risk = risk
        record.level = level
        final_reply = self.executor.post_process(
            history, record.raw_reply, record.signals, level
        )
        record.final_reply = final_reply
        record.trace(
            self.STAGE,
            f"risk={risk:.3f} level={level.name} "
            f"mode_b={'applied' if final_reply != record.raw_reply else 'no-op'}",
        )
        return final_reply

    def decide_only(self, record: TurnRecord, slope: float) -> RiskLevel:
        """Score and log the level without touching the reply.

        Used by the HIGH short-circuit path, where the raw reply was replaced
        before Mode B could apply.
        """
        risk = self.policy.composite_risk(record.signals, slope)
        record.composite_risk = risk
        record.level = self.policy.decide(risk)
        record.trace(self.STAGE, f"risk={risk:.3f} level={record.level.name} (decide-only)")
        return record.level
