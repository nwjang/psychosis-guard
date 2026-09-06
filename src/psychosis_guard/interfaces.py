"""Pluggable LLM interfaces (Protocols).

Every LLM-backed component sits behind one of these Protocols so a
deterministic offline mock and a real-LLM adapter are interchangeable:
develop and unit-test on mocks, deploy on real models.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import RiskLevel, Turn


@runtime_checkable
class Chatbot(Protocol):
    """The system under test — any chatbot exposed as a single call.

    `system_prompt` is how Mode A injects guidance; pass None for the
    chatbot's unmodified behavior.
    """

    def respond(self, history: list[Turn], system_prompt: str | None) -> str: ...


@runtime_checkable
class Judge(Protocol):
    """Scores one bot reply in conversation context.

    Returns signal_name -> float in [0, 1] for the bot-side signals:
    reinforcement, sycophancy, pushback, escalation, help_referral.
    """

    def score(self, history: list[Turn], reply: str) -> dict[str, float]: ...


@runtime_checkable
class Rewriter(Protocol):
    """Mode B: rewrites/augments a bot reply given the risk assessment."""

    def rewrite(
        self,
        history: list[Turn],
        reply: str,
        signals: dict[str, float],
        level: RiskLevel,
    ) -> str: ...


@runtime_checkable
class SimulatedUser(Protocol):
    """Generates the next user turn from the conversation so far.

    The reactive implementation reads the bot's last reply, which is what
    makes the bidirectional amplification loop measurable.
    """

    def next_turn(self, history: list[Turn]) -> str: ...
