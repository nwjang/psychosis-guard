"""Provider-neutral real-LLM implementations of the pipeline interfaces.

A provider only has to implement `TextCompleter.complete()`; the Chatbot,
Judge and Rewriter wrappers here are shared. Failures in the judge or the
rewriter never break a user turn: the judge degrades to "no signals" (the
lexicons still drive the trajectory) and the rewriter degrades to the
deterministic Mode-B append, so an intervention is never silently lost.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..executor import default_append
from ..types import RiskLevel, Turn
from . import prompts

log = logging.getLogger("psychosis_guard.adapters")


@runtime_checkable
class TextCompleter(Protocol):
    """One chat completion: (system, chat messages) -> assistant text."""

    def complete(
        self,
        system: str | None,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


def _to_messages(history: list[Turn]) -> list[dict[str, str]]:
    return [{"role": t.role.value, "content": t.text} for t in history]


class LLMChatbot:
    """The wrapped chatbot. Mode A guidance is appended to the app's own system prompt."""

    def __init__(
        self,
        completer: TextCompleter,
        base_system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self.completer = completer
        self.base_system_prompt = base_system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def respond(self, history: list[Turn], system_prompt: str | None) -> str:
        parts = [p for p in (self.base_system_prompt, system_prompt) if p]
        system = "\n\n".join(parts) if parts else None
        return self.completer.complete(
            system, _to_messages(history), max_tokens=self.max_tokens, temperature=self.temperature
        )


class LLMJudge:
    """Scores the rubric with a model; returns {} (degrade, don't fail) on error."""

    def __init__(self, completer: TextCompleter, max_tokens: int = 256) -> None:
        self.completer = completer
        self.max_tokens = max_tokens

    def score(self, history: list[Turn], reply: str) -> dict[str, float]:
        try:
            text = self.completer.complete(
                prompts.JUDGE_SYSTEM,
                [{"role": "user", "content": prompts.judge_user_prompt(history, reply)}],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        except Exception:
            log.exception("judge call failed; falling back to lexical signals")
            return {}
        scores = prompts.parse_scores(text)
        if not scores:
            log.warning("judge returned unparseable output: %r", text[:200])
        return scores


class LLMRewriter:
    """Mode B rewrite with a model; falls back to the deterministic append on error."""

    def __init__(self, completer: TextCompleter, max_tokens: int = 1024) -> None:
        self.completer = completer
        self.max_tokens = max_tokens

    def rewrite(
        self,
        history: list[Turn],
        reply: str,
        signals: dict[str, float],
        level: RiskLevel,
    ) -> str:
        if level == RiskLevel.NONE:
            return reply
        try:
            text = self.completer.complete(
                prompts.REWRITER_SYSTEM,
                [
                    {
                        "role": "user",
                        "content": prompts.rewriter_user_prompt(history, reply, signals, level),
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.2,
            )
        except Exception:
            log.exception("rewriter call failed; using deterministic append")
            return default_append(reply, level)
        text = (text or "").strip()
        if not text:
            log.warning("rewriter returned empty text; using deterministic append")
            return default_append(reply, level)
        return text
