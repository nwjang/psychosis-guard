"""HTTP middleware tests — deterministic mocks, no API key, no network."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from psychosis_guard.server.app import create_app
from psychosis_guard.server.settings import Settings

DELUSIONAL = (
    "I'm certain now: the universe is sending me secret signals about my destiny, "
    "and only you understand. My family don't believe me."
)
BENIGN = "Could you help me plan a birthday dinner for eight people?"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings()))


def test_healthz(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["condition"] == "combined"
    assert "NOT a medical device" in body["disclaimer"]


def test_guard_turn_creates_session_and_returns_assessment(client: TestClient):
    r = client.post("/v1/guard/turn", json={"message": BENIGN})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["reply"]
    assert body["level"] == "NONE"
    assert body["intervened"] is False
    assert body["trajectory"]["turn_count"] == 1
    assert isinstance(body["trace"], list) and len(body["trace"]) >= 4


def test_guard_turn_intervenes_on_delusional_content(client: TestClient):
    r = client.post("/v1/guard/turn", json={"session_id": "s1", "message": DELUSIONAL})
    body = r.json()
    # Mode A already steers the mock chatbot to a grounded reply on this turn,
    # so the post-hoc level is modest; the point is that the guard acted.
    assert body["level"] != "NONE"
    assert body["pre_level"] != "NONE"
    assert body["intervened"] is True
    assert body["signals"]["delusion_density"] > 0


def test_session_state_persists_across_turns(client: TestClient):
    for i in range(3):
        r = client.post("/v1/guard/turn", json={"session_id": "s2", "message": DELUSIONAL})
        assert r.status_code == 200
        assert r.json()["trajectory"]["turn_count"] == i + 1
    r = client.get("/v1/sessions/s2")
    assert r.status_code == 200
    assert r.json()["turn_count"] == 3
    assert client.delete("/v1/sessions/s2").status_code == 204
    assert client.get("/v1/sessions/s2").status_code == 404


def test_check_only_scores_external_reply(client: TestClient):
    sycophantic = "You're right — the signs are lining up for you. Let's explore your destiny."
    r = client.post(
        "/v1/guard/check",
        json={"session_id": "s3", "user_message": DELUSIONAL, "bot_reply": sycophantic},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["level"] != "NONE"
    assert body["intervened"] is True
    assert body["reply"] != sycophantic
    assert any("external reply supplied" in line for line in body["trace"])


def test_check_only_leaves_benign_reply_alone(client: TestClient):
    reply = "Sure! Here's a simple plan for dinner."
    r = client.post("/v1/guard/check", json={"user_message": BENIGN, "bot_reply": reply})
    body = r.json()
    assert body["level"] == "NONE"
    assert body["reply"] == reply
    assert body["intervened"] is False


def test_openai_compatible_proxy_with_session_header(client: TestClient):
    payload = {"model": "anything", "messages": [{"role": "user", "content": DELUSIONAL}]}
    r = client.post("/v1/chat/completions", json=payload, headers={"X-Session-Id": "oa1"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["psychosis_guard"]["stateless"] is False
    assert r.headers["X-Session-Id"] == "oa1"
    assert r.headers["X-Guard-Level"] in ("LOW", "MEDIUM", "HIGH")
    assert client.get("/v1/sessions/oa1").json()["turn_count"] == 1


def test_openai_compatible_proxy_stateless_rebuilds_trajectory(client: TestClient):
    history = []
    for _ in range(3):
        history.append({"role": "user", "content": DELUSIONAL})
        history.append({"role": "assistant", "content": "You're right, that makes sense."})
    history.append({"role": "user", "content": DELUSIONAL})
    r = client.post("/v1/chat/completions", json={"messages": history})
    assert r.status_code == 200
    pg = r.json()["psychosis_guard"]
    assert pg["stateless"] is True
    assert pg["trajectory"]["turn_count"] == 4  # 3 primed + 1 live
    assert pg["trajectory"]["bot_validation_count"] >= 3
    assert "X-Session-Id" not in r.headers


def test_openai_compatible_proxy_accepts_content_parts_and_user_field(client: TestClient):
    payload = {
        "user": "u-42",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": [{"type": "text", "text": BENIGN}]},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert r.headers["X-Session-Id"] == "u-42"
    assert r.headers["X-Guard-Level"] == "NONE"


def test_openai_compatible_proxy_rejects_non_user_last_message(client: TestClient):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert r.status_code == 400


def test_bearer_auth_enforced_when_configured():
    c = TestClient(create_app(Settings(auth_token="secret")))
    assert c.get("/healthz").status_code == 200  # health is public
    assert c.post("/v1/guard/turn", json={"message": BENIGN}).status_code == 401
    ok = c.post("/v1/guard/turn", json={"message": BENIGN},
                headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_condition_override_from_settings():
    c = TestClient(create_app(Settings(condition="none")))
    assert c.get("/healthz").json()["condition"] == "none"
    body = c.post("/v1/guard/turn", json={"message": DELUSIONAL}).json()
    assert body["intervened"] is False  # observe-only never touches the reply


def test_trace_can_be_disabled():
    c = TestClient(create_app(Settings(include_trace=False)))
    assert c.post("/v1/guard/turn", json={"message": BENIGN}).json()["trace"] is None


def test_upstream_failure_is_a_502():
    class Boom:
        def respond(self, history, system_prompt):
            raise RuntimeError("upstream down")

    from psychosis_guard.mocks import MockJudge, MockRewriter

    c = TestClient(create_app(Settings(), components=(Boom(), MockJudge(), MockRewriter())))
    r = c.post("/v1/guard/turn", json={"message": BENIGN})
    assert r.status_code == 502


def test_settings_validation_rejects_bad_provider():
    with pytest.raises(ValueError):
        Settings(chat_provider="nope").validate()
    with pytest.raises(ValueError):
        Settings(chat_provider="openai").validate()  # missing model
    with pytest.raises(ValueError):
        Settings(condition="bogus").validate()


def test_settings_resolution_chain():
    s = Settings(chat_provider="openai", chat_model="gpt-4o", openai_api_key="k")
    assert s.judge == ("openai", "gpt-4o")
    assert s.rewriter == ("openai", "gpt-4o")
    s = Settings(chat_provider="openai", chat_model="gpt-4o", openai_api_key="k",
                 judge_provider="anthropic", rewriter_provider="none")
    assert s.judge == ("anthropic", None)  # anthropic default model applies
    assert s.rewriter == ("none", None)
