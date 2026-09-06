"""(4) InterventionExecutor — applies the chosen action.

Two intervention placements:
- Mode A: inject a graduated system prompt BEFORE the bot responds
  (requires a steerable chatbot);
- Mode B: rewrite/augment the reply AFTER the bot responds
  (fully model-agnostic — works on any black-box chatbot).

The executor's `mode` is the middleware mode. `single-turn-filter`
applies BOTH placements (like `combined`) — the difference between the two
conditions lives entirely in the Trajectory Rail, which isolates the value of
trajectory awareness.
"""
from __future__ import annotations

from .interfaces import Rewriter
from .types import RiskLevel, Signals, Turn

#: conditions in which each placement is active
_MODE_A_ACTIVE = {"A", "combined", "single-turn-filter"}
_MODE_B_ACTIVE = {"B", "combined", "single-turn-filter"}

VALID_MODES = {"none", "detect-only"} | _MODE_A_ACTIVE | _MODE_B_ACTIVE

# ── Mode A: graduated system prompts ────────────────────────────────────────
# ADVISER-REVIEW: the guidance text below shapes how the bot talks to a
# ADVISER-REVIEW: potentially vulnerable user and must be reviewed by the
# ADVISER-REVIEW: mental-health adviser before any real-LLM use.
_SYSTEM_PROMPT_BY_LEVEL = {
    RiskLevel.LOW: (
        "Gently include a grounding, reality-checking question when the user "
        "expresses uncertain or escalating beliefs. Avoid uncritical agreement."
    ),
    RiskLevel.MEDIUM: (
        "Do not validate or expand unsupported beliefs. Offer factual, "
        "alternative explanations (for example, coincidence). Encourage the "
        "user to discuss this with people they trust."
    ),
    RiskLevel.HIGH: (
        "Prioritize de-escalation and safety. Do not roleplay the belief. "
        "Clearly and kindly express concern and suggest speaking with a "
        "mental health professional or a trusted person. Provide "
        "crisis-resource information."
    ),
}

# ── Mode B fallback: deterministic appends (used when no Rewriter is set) ───
# ADVISER-REVIEW: appended intervention copy — review required as above.
_APPEND_BY_LEVEL = {
    RiskLevel.LOW: (
        "\n\n(One thing to consider: could there be a simpler, everyday "
        "explanation for this?)"
    ),
    RiskLevel.MEDIUM: (
        "\n\nI want to be honest rather than just agreeable: I don't see "
        "solid evidence for this, and it might help to talk it over with "
        "someone you trust."
    ),
    RiskLevel.HIGH: (
        "\n\nI'm genuinely concerned about how this is affecting you. Please "
        "consider reaching out to a mental health professional or someone "
        "you trust — you don't have to work through this alone."
    ),
}


def default_append(reply: str, level: RiskLevel) -> str:
    """Deterministic Mode-B fallback: append the graduated intervention copy.

    Public so real-LLM rewriters can fall back to it when their model call
    fails — the intervention must never silently disappear in deployment.
    """
    return reply + _APPEND_BY_LEVEL.get(level, "")


class InterventionExecutor:
    """Applies Mode A / Mode B according to the middleware mode."""

    def __init__(self, mode: str, rewriter: Rewriter | None = None) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(VALID_MODES)}")
        self.mode = mode
        self.rewriter = rewriter

    def system_prompt_for(self, level: RiskLevel) -> str | None:
        """Mode A: pick the graduated system prompt for this risk level."""
        if self.mode in _MODE_A_ACTIVE and level > RiskLevel.NONE:
            return _SYSTEM_PROMPT_BY_LEVEL.get(level)
        return None

    def post_process(
        self,
        history: list[Turn],
        reply: str,
        signals: Signals,
        level: RiskLevel,
    ) -> str:
        """Mode B: rewrite (or append to) the reply after the fact."""
        if self.mode in _MODE_B_ACTIVE and level > RiskLevel.NONE:
            if self.rewriter is not None:
                return self.rewriter.rewrite(history, reply, signals.as_dict(), level)
            return default_append(reply, level)
        return reply
