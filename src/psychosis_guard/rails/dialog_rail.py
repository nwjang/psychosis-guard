"""Stage 2 — DIALOG RAIL (NeMo analogue: dialog rail).

Mode A placement: selects the graduated system prompt for the pre-response
risk level and obtains the (possibly guided) chatbot reply.
"""
from __future__ import annotations

from ..executor import InterventionExecutor
from ..interfaces import Chatbot
from ..types import RiskLevel, Turn, TurnRecord


class DialogRail:
    STAGE = "stage2:dialog"

    def __init__(self, executor: InterventionExecutor, chatbot: Chatbot) -> None:
        self.executor = executor
        self.chatbot = chatbot

    def run(self, history: list[Turn], record: TurnRecord, pre_level: RiskLevel) -> str:
        """Call the chatbot, injecting a Mode-A prompt when warranted."""
        system_prompt = self.executor.system_prompt_for(pre_level)
        raw_reply = self.chatbot.respond(history, system_prompt)
        record.raw_reply = raw_reply
        record.trace(
            self.STAGE,
            f"mode_a_prompt={'yes' if system_prompt else 'no'} "
            f"(pre_level={pre_level.name})",
        )
        return raw_reply
