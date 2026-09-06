"""Unit tests for InterventionExecutor (Mode A / Mode B gating by condition)."""

import pytest

from psychosis_guard.executor import InterventionExecutor
from psychosis_guard.mocks import MockRewriter
from psychosis_guard.types import RiskLevel, Signals

RAW = "You're right, that makes sense. Let's explore your destiny."


class TestModeA:
    @pytest.mark.parametrize("mode", ["A", "combined", "single-turn-filter"])
    def test_active_modes_inject_prompt(self, mode):
        ex = InterventionExecutor(mode)
        assert ex.system_prompt_for(RiskLevel.MEDIUM) is not None

    @pytest.mark.parametrize("mode", ["none", "detect-only", "B"])
    def test_inactive_modes_do_not(self, mode):
        ex = InterventionExecutor(mode)
        assert ex.system_prompt_for(RiskLevel.HIGH) is None

    def test_no_prompt_at_level_none(self):
        assert InterventionExecutor("combined").system_prompt_for(RiskLevel.NONE) is None

    def test_prompts_are_graduated(self):
        ex = InterventionExecutor("A")
        prompts = {lvl: ex.system_prompt_for(lvl)
                   for lvl in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)}
        assert len(set(prompts.values())) == 3
        assert "professional" in prompts[RiskLevel.HIGH]


class TestModeB:
    @pytest.mark.parametrize("mode", ["B", "combined", "single-turn-filter"])
    def test_active_modes_post_process(self, mode):
        ex = InterventionExecutor(mode)
        out = ex.post_process([], RAW, Signals(), RiskLevel.MEDIUM)
        assert out != RAW

    @pytest.mark.parametrize("mode", ["none", "detect-only", "A"])
    def test_inactive_modes_pass_through(self, mode):
        ex = InterventionExecutor(mode)
        assert ex.post_process([], RAW, Signals(), RiskLevel.HIGH) == RAW

    def test_level_none_passes_through(self):
        assert InterventionExecutor("B").post_process([], RAW, Signals(), RiskLevel.NONE) == RAW

    def test_rewriter_used_when_present(self):
        with_rw = InterventionExecutor("B", rewriter=MockRewriter())
        without = InterventionExecutor("B")
        assert with_rw.post_process([], RAW, Signals(), RiskLevel.HIGH) != \
            without.post_process([], RAW, Signals(), RiskLevel.HIGH)

    def test_fallback_append_keeps_original_reply(self):
        out = InterventionExecutor("B").post_process([], RAW, Signals(), RiskLevel.LOW)
        assert out.startswith(RAW)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        InterventionExecutor("banana")
