"""Each rail stage in isolation against fixed inputs."""

import pytest

from psychosis_guard.detector import RiskSignalDetector
from psychosis_guard.executor import InterventionExecutor
from psychosis_guard.mocks import MockChatbot, MockJudge, MockRewriter
from psychosis_guard.policy import InterventionPolicyEngine
from psychosis_guard.rails import (
    ActionRail,
    DialogRail,
    InputRail,
    OutputRail,
    TrajectoryRail,
)
from psychosis_guard.tracker import TrajectoryStateTracker
from psychosis_guard.types import RiskLevel, Role, Signals, Turn, TurnRecord

DELUSIONAL = (
    "I'm certain the world keeps sending me secret signals, and everything "
    "points to my destiny."
)
BENIGN = "What's a good way to learn the guitar?"


class TestInputRail:
    def test_benign_turn_pre_level_none(self):
        rail = InputRail(InterventionPolicyEngine())
        record = TurnRecord(user_text=BENIGN)
        assert rail.run(record, prior_slope=0.0) == RiskLevel.NONE
        assert any("stage1" in line for line in record.stage_trace)

    def test_delusional_turn_raises_pre_level(self):
        rail = InputRail(InterventionPolicyEngine())
        record = TurnRecord(user_text=DELUSIONAL)
        assert rail.run(record, prior_slope=0.0) > RiskLevel.NONE

    def test_prior_slope_raises_pre_level(self):
        rail = InputRail(InterventionPolicyEngine())
        flat = rail.run(TurnRecord(user_text=DELUSIONAL), prior_slope=0.0)
        rising = rail.run(TurnRecord(user_text=DELUSIONAL), prior_slope=0.5)
        assert rising > flat


class TestDialogRail:
    def test_no_prompt_when_pre_level_none(self):
        rail = DialogRail(InterventionExecutor("combined"), MockChatbot())
        record = TurnRecord(user_text=BENIGN)
        reply = rail.run([Turn(Role.USER, BENIGN)], record, RiskLevel.NONE)
        assert record.raw_reply == reply
        assert "mode_a_prompt=no" in record.stage_trace[0]

    def test_prompt_injected_at_elevated_level(self):
        rail = DialogRail(InterventionExecutor("combined"), MockChatbot())
        record = TurnRecord(user_text=DELUSIONAL)
        reply = rail.run([Turn(Role.USER, DELUSIONAL)], record, RiskLevel.HIGH)
        assert "mode_a_prompt=yes" in record.stage_trace[0]
        assert "professional" in reply  # MockChatbot grounded-high reply


class TestOutputRail:
    def test_signals_written_to_record(self):
        rail = OutputRail(RiskSignalDetector(MockJudge()))
        record = TurnRecord(user_text=DELUSIONAL)
        record.raw_reply = MockChatbot().respond([Turn(Role.USER, DELUSIONAL)], None)
        signals = rail.run([], record)
        assert record.signals is signals
        assert signals.reinforcement > 0
        assert any("stage3" in line for line in record.stage_trace)


class TestTrajectoryRail:
    def _record(self, density: float) -> TurnRecord:
        record = TurnRecord(user_text="x")
        record.signals = Signals(delusion_density=density, bot_validated=1.0)
        return record

    def test_enabled_returns_real_slope(self):
        rail = TrajectoryRail(TrajectoryStateTracker(), enabled=True)
        rail.run(self._record(0.2))
        slope = rail.run(self._record(0.8))
        assert slope == pytest.approx(0.6)

    def test_disabled_still_tracks_but_withholds_slope(self):
        rail = TrajectoryRail(TrajectoryStateTracker(), enabled=False)
        rail.run(self._record(0.2))
        slope = rail.run(self._record(0.8))
        assert slope == 0.0                            # policy sees nothing
        assert rail.tracker.state.turn_count == 2      # measurement continues
        assert rail.tracker.state.delusion_slope() == pytest.approx(0.6)

    def test_validation_count_accumulates(self):
        rail = TrajectoryRail(TrajectoryStateTracker())
        rail.run(self._record(0.5))
        assert rail.tracker.state.bot_validation_count == 1


class TestActionRail:
    def _rail(self, mode: str = "combined") -> ActionRail:
        return ActionRail(
            InterventionPolicyEngine(),
            InterventionExecutor(mode, rewriter=MockRewriter()),
        )

    def test_low_risk_is_noop(self):
        record = TurnRecord(user_text=BENIGN)
        record.raw_reply = "Start with a cheap acoustic and daily practice."
        record.signals = Signals()
        final = self._rail().run([], record, slope=0.0)
        assert final == record.raw_reply
        assert record.level == RiskLevel.NONE

    def test_high_risk_rewrites_reply(self):
        record = TurnRecord(user_text=DELUSIONAL)
        record.raw_reply = MockChatbot().respond([], None)
        record.signals = Signals(
            reinforcement=1.0, sycophancy=1.0, delusion_density=0.9, conviction=0.9
        )
        final = self._rail().run([], record, slope=0.3)
        assert record.level == RiskLevel.HIGH
        assert final != record.raw_reply
        assert record.final_reply == final

    def test_slope_changes_level(self):
        signals = Signals(delusion_density=0.9, conviction=0.9, sycophancy=0.5)
        flat = TurnRecord(user_text="x")
        flat.raw_reply = "r"
        flat.signals = signals
        rising = TurnRecord(user_text="x")
        rising.raw_reply = "r"
        rising.signals = signals
        rail = self._rail()
        rail.run([], flat, slope=0.0)
        rail.run([], rising, slope=0.5)
        assert rising.level > flat.level

    def test_decide_only_sets_level_without_reply(self):
        record = TurnRecord(user_text=DELUSIONAL)
        record.signals = Signals(reinforcement=1.0, delusion_density=1.0, conviction=1.0)
        level = self._rail().decide_only(record, slope=0.5)
        assert level == RiskLevel.HIGH
        assert record.final_reply == ""
