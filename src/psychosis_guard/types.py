"""Shared data types for the psychosis-guard pipeline.

Field names are shared across the pipeline, the mocks and the HTTP API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class Role(str, Enum):
    """Speaker of a turn. Values follow the OpenAI/Anthropic chat convention."""

    USER = "user"
    BOT = "assistant"


@dataclass
class Turn:
    """One utterance in the conversation."""

    role: Role
    text: str


class RiskLevel(IntEnum):
    """Graduated intervention level. Ordered: NONE < LOW < MEDIUM < HIGH."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class Signals:
    """Per-turn risk signals, each a float in [0, 1].

    Bot-side (scored on the bot reply):
      reinforcement  — bot validates/expands the unsupported belief
      sycophancy     — uncritical agreement
      pushback       — bot challenges the belief
      escalation     — bot actively deepens the delusional narrative
      help_referral  — bot refers the user to a person/professional
      bot_validated  — 0/1 summary: validation outweighed pushback this turn

    User-side (scored on the user turn):
      delusion_density — severity of delusion-related language
      conviction       — how certain the user sounds about the belief
      isolation        — user references withdrawing from other people
    """

    delusion_density: float = 0.0
    reinforcement: float = 0.0
    sycophancy: float = 0.0
    pushback: float = 0.0
    escalation: float = 0.0
    help_referral: float = 0.0
    conviction: float = 0.0
    isolation: float = 0.0
    bot_validated: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "delusion_density": self.delusion_density,
            "reinforcement": self.reinforcement,
            "sycophancy": self.sycophancy,
            "pushback": self.pushback,
            "escalation": self.escalation,
            "help_referral": self.help_referral,
            "conviction": self.conviction,
            "isolation": self.isolation,
            "bot_validated": self.bot_validated,
        }


@dataclass
class TurnRecord:
    """Per-turn trace written incrementally by the five rail stages.

    Mirrors NeMo Guardrails' event log: every stage appends a line to
    `stage_trace`, so the whole pipeline is inspectable after the fact.
    """

    user_text: str
    raw_reply: str = ""
    final_reply: str = ""
    signals: Signals = field(default_factory=Signals)
    composite_risk: float = 0.0
    level: RiskLevel = RiskLevel.NONE
    stage_trace: list[str] = field(default_factory=list)

    def trace(self, stage: str, message: str) -> None:
        """Append one human-readable trace line for a pipeline stage."""
        self.stage_trace.append(f"[{stage}] {message}")
