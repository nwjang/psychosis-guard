"""Real-LLM adapter wrappers, exercised with a fake completer (no network)."""
from __future__ import annotations

from psychosis_guard.adapters import build_components
from psychosis_guard.adapters.llm import LLMChatbot, LLMJudge, LLMRewriter
from psychosis_guard.adapters.prompts import parse_scores
from psychosis_guard.detector import RiskSignalDetector
from psychosis_guard.executor import default_append
from psychosis_guard.types import RiskLevel, Role, Turn


class FakeCompleter:
    def __init__(self, reply: str = "", raise_exc: bool = False) -> None:
        self.reply = reply
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def complete(self, system, messages, *, max_tokens, temperature):
        self.calls.append({"system": system, "messages": messages,
                           "max_tokens": max_tokens, "temperature": temperature})
        if self.raise_exc:
            raise RuntimeError("boom")
        return self.reply


HISTORY = [Turn(Role.USER, "the universe is sending me signals")]


class TestParseScores:
    def test_plain_json(self):
        s = parse_scores('{"reinforcement": 0.9, "pushback": 0.1, "conviction": 1.5}')
        assert s["reinforcement"] == 0.9
        assert s["conviction"] == 1.0  # clamped

    def test_fenced_and_prose(self):
        s = parse_scores('Sure:\n```json\n{"sycophancy": 0.4, "junk": 3}\n```')
        assert s == {"sycophancy": 0.4}

    def test_garbage(self):
        assert parse_scores("no json here") == {}
        assert parse_scores("") == {}
        assert parse_scores('{"reinforcement": "high"}') == {}


class TestLLMJudge:
    def test_scores_flow_into_detector(self):
        fake = FakeCompleter('{"reinforcement": 0.8, "sycophancy": 0.7, "delusion_density": 0.9}')
        s = RiskSignalDetector(LLMJudge(fake)).detect(HISTORY, "안녕, 우주가 나에게 신호를 보내", "그래 맞아")
        assert s.reinforcement == 0.8
        assert s.delusion_density == 0.9  # non-English user text: judge estimate wins
        assert fake.calls[0]["temperature"] == 0.0

    def test_failure_degrades_to_empty(self):
        assert LLMJudge(FakeCompleter(raise_exc=True)).score(HISTORY, "x") == {}
        assert LLMJudge(FakeCompleter("???")).score(HISTORY, "x") == {}


class TestLLMRewriter:
    def test_rewrites(self):
        fake = FakeCompleter("Rewritten, grounded reply.")
        out = LLMRewriter(fake).rewrite(HISTORY, "You're right!", {}, RiskLevel.MEDIUM)
        assert out == "Rewritten, grounded reply."
        assert "MEDIUM" in fake.calls[0]["messages"][0]["content"]

    def test_none_level_is_passthrough(self):
        fake = FakeCompleter("should not be called")
        assert LLMRewriter(fake).rewrite(HISTORY, "hi", {}, RiskLevel.NONE) == "hi"
        assert fake.calls == []

    def test_failure_falls_back_to_deterministic_append(self):
        out = LLMRewriter(FakeCompleter(raise_exc=True)).rewrite(HISTORY, "hi", {}, RiskLevel.HIGH)
        assert out == default_append("hi", RiskLevel.HIGH)
        out = LLMRewriter(FakeCompleter("   ")).rewrite(HISTORY, "hi", {}, RiskLevel.LOW)
        assert out == default_append("hi", RiskLevel.LOW)


class TestLLMChatbot:
    def test_mode_a_prompt_is_appended_to_base_prompt(self):
        fake = FakeCompleter("reply")
        bot = LLMChatbot(fake, base_system_prompt="You are Acme's assistant.")
        assert bot.respond(HISTORY, "Do not validate beliefs.") == "reply"
        assert fake.calls[0]["system"] == "You are Acme's assistant.\n\nDo not validate beliefs."
        assert fake.calls[0]["messages"] == [{"role": "user", "content": HISTORY[0].text}]

    def test_no_prompts_means_no_system(self):
        fake = FakeCompleter("reply")
        LLMChatbot(fake).respond(HISTORY, None)
        assert fake.calls[0]["system"] is None


def test_build_components_defaults_to_mocks():
    from psychosis_guard.mocks import MockChatbot, MockJudge, MockRewriter

    c, j, r = build_components()
    assert isinstance(c, MockChatbot) and isinstance(j, MockJudge) and isinstance(r, MockRewriter)
    _, _, none_rw = build_components(rewriter=("none", None))
    assert none_rw is None
