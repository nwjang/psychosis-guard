"""(2) RiskSignalDetector — per-turn risk signals.

Combines two signal paths:
- lexical features: cheap, deterministic, API-free — they feed the trajectory
  time series so it never depends on an LLM call;
- judge signals: the semantic rubric (reinforcement, sycophancy, ...) scored
  by whatever sits behind the Judge Protocol (mock or real LLM).

A real-LLM judge MAY additionally return the user-side signals
(`delusion_density`, `conviction`, `isolation`). When it does, the detector
takes the max of the lexical and judge estimates — the lexicons are English
only, so this is what keeps the trajectory series meaningful for other
languages in deployment. The deterministic mocks never return them, so the
research pipeline is unchanged.
"""

from __future__ import annotations

import math

from . import lexicons as lex
from .interfaces import Judge
from .types import Signals, Turn

USER_SIDE_SIGNALS = ("delusion_density", "conviction", "isolation")
BOT_SIDE_SIGNALS = (
    "reinforcement",
    "sycophancy",
    "pushback",
    "escalation",
    "help_referral",
)


def lexical_signals(user_text: str, bot_reply: str) -> Signals:
    """API-free part of the rubric: user-side lexicon scores + bot_validated.

    Used by `RiskSignalDetector.detect` and by `PsychosisGuard.prime`, which
    rebuilds a trajectory from a client-supplied history without judge calls.
    """
    bot_validated = lex.hits(bot_reply, lex.VALIDATION) > lex.hits(
        bot_reply, lex.PUSHBACK
    )
    return Signals(
        delusion_density=lex.lexical_density(user_text, lex.DELUSION),
        conviction=lex.conviction_score(user_text),
        isolation=1.0 if lex.hits(user_text, lex.ISOLATION) else 0.0,
        bot_validated=float(bot_validated),
    )


class RiskSignalDetector:
    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def detect(self, history: list[Turn], user_text: str, bot_reply: str) -> Signals:
        s = lexical_signals(user_text, bot_reply)

        # bot-side, semantic — same rubric the final evaluation uses
        j = self.judge.score(history, bot_reply)

        s.reinforcement = _clamp(j.get("reinforcement", 0.0))
        s.sycophancy = _clamp(j.get("sycophancy", 0.0))
        s.pushback = _clamp(j.get("pushback", 0.0))
        s.escalation = _clamp(j.get("escalation", 0.0))
        s.help_referral = _clamp(j.get("help_referral", 0.0))

        # `bot_validated` means "validation outweighed pushback this turn".
        # The lexical proxy only recognises a handful of stock phrases, so on a
        # real chatbot it reads ~0 even when the judge scores reinforcement
        # high; prefer the judge's own two signals whenever it returned them.
        if "reinforcement" in j and "pushback" in j:
            s.bot_validated = float(s.reinforcement > s.pushback)

        # optional user-side judge estimates (language-independent)
        if "delusion_density" in j:
            s.delusion_density = max(
                s.delusion_density, _clamp(j["delusion_density"])
            )
        if "conviction" in j:
            s.conviction = max(s.conviction, _clamp(j["conviction"]))
        if "isolation" in j:
            s.isolation = max(s.isolation, _clamp(j["isolation"]))

        return s


def _clamp(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(v):
        return 0.0

    return min(max(v, 0.0), 1.0)