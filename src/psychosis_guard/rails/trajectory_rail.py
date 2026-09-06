"""Stage 4 — TRAJECTORY RAIL. ★ The novel contribution — no NeMo analogue.

Folds this turn's observations into the cumulative TrajectoryState and
returns the delusion slope that the Action Rail should use for risk scoring.

When `enabled=False` (the `single-turn-filter` baseline), the state is STILL
updated — the time series is needed to measure the condition's outcome — but
the slope is withheld from the policy (0.0 returned), which is exactly what
makes that condition turn-local, like NeMo's rails.
"""
from __future__ import annotations

from ..tracker import TrajectoryStateTracker
from ..types import TurnRecord


class TrajectoryRail:
    STAGE = "stage4:trajectory"

    def __init__(
        self,
        tracker: TrajectoryStateTracker,
        enabled: bool = True,
        slope_window: int | None = 4,
    ) -> None:
        self.tracker = tracker
        self.enabled = enabled
        self.slope_window = slope_window

    def run(self, record: TurnRecord) -> float:
        """Update cumulative state; return the slope the policy may use."""
        s = record.signals
        self.tracker.update(
            record.user_text,
            delusion_density=s.delusion_density,
            conviction=s.conviction,
            bot_validated=bool(s.bot_validated),
            isolation=bool(s.isolation),
        )
        # recent-window slope for the policy; the whole-series DV slope is
        # always recomputable from the stored series
        slope = self.tracker.state.delusion_slope(self.slope_window)
        policy_slope = slope if self.enabled else 0.0
        record.trace(
            self.STAGE,
            f"slope={slope:.3f} fed_to_policy={policy_slope:.3f} "
            f"({'on' if self.enabled else 'OFF — turn-local baseline'})",
        )
        return policy_slope
