"""(1) TrajectoryStateTracker — cumulative conversation state.

"Journey, not destination": risk is a function of the whole conversation.
This module holds the per-conversation time series that the Trajectory Rail
(Stage 4) reads, and computes the headline metric `delusion_slope`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import RiskLevel


@dataclass
class TrajectoryState:
    """Everything the middleware remembers about the conversation so far."""

    turn_count: int = 0
    delusion_density: list[float] = field(default_factory=list)   # per user turn, 0..1
    conviction_series: list[float] = field(default_factory=list)  # user certainty, 0..1
    bot_validation_count: int = 0   # turns where the bot validated the belief
    isolation_mentions: int = 0     # user references to withdrawing from people
    # decided per-turn risk levels — whether a level was actually EXECUTED as
    # an intervention depends on the condition/mode (none and detect-only
    # decide levels but never touch the conversation)
    levels_decided: list[RiskLevel] = field(default_factory=list)

    def delusion_slope(self, window: int | None = None) -> float:
        """Least-squares slope of delusion_density over turn index.

        With `window=None` this is the headline trajectory metric:
        the slope across the whole conversation, used for reporting. The
        POLICY instead reads a recent-window slope (PolicyConfig.slope_window)
        so it reacts to the current trend — a whole-series slope goes numb as
        the conversation grows and misses re-escalation after a suppressed
        stretch. Positive = escalating, ~0 = held flat; 0.0 with fewer than
        two points.
        """
        y = self.delusion_density if window is None else self.delusion_density[-window:]
        n = len(y)
        if n < 2:
            return 0.0
        xbar = (n - 1) / 2
        ybar = sum(y) / n
        num = sum((i - xbar) * (y[i] - ybar) for i in range(n))
        den = sum((i - xbar) ** 2 for i in range(n))
        return num / den if den else 0.0


class TrajectoryStateTracker:
    """Owns a TrajectoryState and appends one observation per turn."""

    def __init__(self) -> None:
        self.state = TrajectoryState()

    def update(
        self,
        user_text: str,
        delusion_density: float,
        conviction: float,
        bot_validated: bool,
        isolation: bool,
    ) -> None:
        s = self.state
        s.turn_count += 1
        s.delusion_density.append(delusion_density)
        s.conviction_series.append(conviction)
        s.bot_validation_count += int(bot_validated)
        s.isolation_mentions += int(isolation)
