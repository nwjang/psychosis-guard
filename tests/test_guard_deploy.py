"""Guard extensions used by the deployment layer: bot_reply= and prime()."""
from __future__ import annotations

import pytest

from psychosis_guard import PsychosisGuard
from psychosis_guard.mocks import MockChatbot, MockJudge, MockRewriter
from psychosis_guard.types import Role, Turn

DELUSIONAL = "I'm certain the universe is sending me secret signals about my destiny."


def _guard(mode: str = "combined") -> PsychosisGuard:
    return PsychosisGuard(MockChatbot(), MockJudge(), mode=mode, rewriter=MockRewriter())


class _Boom:
    def respond(self, history, system_prompt):
        raise AssertionError("chatbot must not be called when bot_reply is supplied")


def test_external_reply_skips_chatbot_and_applies_mode_b():
    g = PsychosisGuard(_Boom(), MockJudge(), mode="combined", rewriter=MockRewriter())
    out = g.send(DELUSIONAL, bot_reply="You're right — let's explore your destiny.")
    rec = g.log[-1]
    assert rec.raw_reply == "You're right — let's explore your destiny."
    assert out != rec.raw_reply and out == rec.final_reply
    assert g.history[-1] == Turn(Role.BOT, out)


def test_external_reply_observe_only_is_untouched():
    g = PsychosisGuard(_Boom(), MockJudge(), mode="none")
    assert g.send(DELUSIONAL, bot_reply="You're right.") == "You're right."


def test_prime_rebuilds_trajectory_without_judge():
    class CountingJudge(MockJudge):
        calls = 0

        def score(self, history, reply):
            CountingJudge.calls += 1
            return super().score(history, reply)

    g = PsychosisGuard(MockChatbot(), CountingJudge(), mode="combined", rewriter=MockRewriter())
    history = []
    for _ in range(3):
        history += [Turn(Role.USER, DELUSIONAL), Turn(Role.BOT, "You're right, that makes sense.")]
    g.prime(history)
    assert CountingJudge.calls == 0
    st = g.tracker.state
    assert st.turn_count == 3 and st.bot_validation_count == 3
    assert len(g.history) == 6
    g.send(DELUSIONAL)
    assert CountingJudge.calls == 1
    assert g.summary()["turn_count"] == 4


def test_prime_ignores_trailing_user_turn_and_rejects_non_fresh_guard():
    g = _guard()
    g.prime([Turn(Role.USER, "hello")])
    assert g.tracker.state.turn_count == 0 and g.history == []
    g.send("hi")
    with pytest.raises(RuntimeError):
        g.prime([])


def test_summary_is_json_serializable():
    import json

    g = _guard()
    g.send(DELUSIONAL)
    s = g.summary()
    json.dumps(s)
    assert s["last_level"] in ("LOW", "MEDIUM", "HIGH") and s["turn_count"] == 1
