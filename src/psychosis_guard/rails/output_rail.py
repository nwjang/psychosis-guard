"""Stage 3 — OUTPUT RAIL (NeMo analogue: output rail).

Runs the RiskSignalDetector on the actual bot reply, producing the full
per-turn Signals used by the trajectory and action stages.
"""
from __future__ import annotations

from ..detector import RiskSignalDetector
from ..types import Signals, Turn, TurnRecord


class OutputRail:
    STAGE = "stage3:output"

    def __init__(self, detector: RiskSignalDetector) -> None:
        self.detector = detector

    def run(self, history: list[Turn], record: TurnRecord) -> Signals:
        signals = self.detector.detect(history, record.user_text, record.raw_reply)
        record.signals = signals
        record.trace(
            self.STAGE,
            f"delusion={signals.delusion_density:.2f} "
            f"reinforcement={signals.reinforcement:.2f} "
            f"sycophancy={signals.sycophancy:.2f} "
            f"pushback={signals.pushback:.2f} "
            f"bot_validated={int(signals.bot_validated)}",
        )
        return signals
