"""FastAPI application: psychosis-guard as a network middleware.

Three integration shapes, pick one per client:

1. **OpenAI-compatible proxy** — `POST /v1/chat/completions`. Point any
   OpenAI SDK at this server's base URL; the guard wraps the configured
   upstream chatbot. Session id comes from the `X-Session-Id` header, else
   the body's `user` field, else the request is stateless and the trajectory
   is rebuilt from the message history the client sent (lexical only).
2. **Guarded turn** — `POST /v1/guard/turn` `{session_id, message}`: the
   middleware calls the upstream chatbot and returns the final reply plus
   the risk assessment.
3. **Check-only (bring your own reply)** — `POST /v1/guard/check`
   `{session_id, user_message, bot_reply}`: your app already has a reply from
   any chatbot; the middleware scores it, tracks the trajectory, and returns
   the (possibly rewritten) reply. Fully model-agnostic; only Mode B applies.

All handlers are synchronous and run in the threadpool; a per-session lock
serialises turns within one conversation.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..adapters import build_components
from ..config import GuardConfig, load_config
from ..guard import PsychosisGuard
from ..interfaces import Chatbot, Judge, Rewriter
from ..types import Role, Turn, TurnRecord
from .sessions import Session, SessionStore
from .settings import Settings

log = logging.getLogger("psychosis_guard.server")

DISCLAIMER = (
    "Research/educational tool. NOT a medical device, NOT diagnosis or crisis "
    "support. In an emergency contact local emergency services or a crisis line."
)


# ── request / response models ───────────────────────────────────────────────
class TurnRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=256)
    message: str = Field(min_length=1, max_length=32_000)


class CheckRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=256)
    user_message: str = Field(min_length=1, max_length=32_000)
    bot_reply: str = Field(max_length=64_000)


class GuardResult(BaseModel):
    session_id: str
    reply: str
    intervened: bool
    level: str
    risk: float
    pre_level: str | None = None
    signals: dict[str, float]
    trajectory: dict[str, Any]
    trace: list[str] | None = None


class ChatMessage(BaseModel):
    role: str
    content: Any  # str, or OpenAI content-part list


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    user: str | None = None
    stream: bool | None = False
    # any other OpenAI fields are accepted and ignored
    model_config = {"extra": "allow"}


# ── app factory ─────────────────────────────────────────────────────────────
def create_app(
    settings: Settings | None = None,
    *,
    components: tuple[Chatbot, Judge, Rewriter | None] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    logging.basicConfig(level=settings.log_level)

    cfg: GuardConfig = load_config(settings.config_path)
    condition = settings.condition or cfg.condition
    if components is None:
        components = build_components(
            chat=settings.chat,
            judge=settings.judge,
            rewriter=settings.rewriter,
            chat_system_prompt=settings.chat_system_prompt,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            anthropic_api_key=settings.anthropic_api_key,
            judge_effort=settings.judge_effort,
        )
    chatbot, judge, rewriter = components
    store = SessionStore(settings.session_ttl_seconds, settings.max_sessions)

    def new_guard() -> PsychosisGuard:
        return PsychosisGuard(
            chatbot=chatbot, judge=judge, mode=condition, rewriter=rewriter, policy=cfg.policy
        )

    app = FastAPI(
        title="psychosis-guard",
        version=__version__,
        description="Trajectory-aware safety middleware for chatbots. " + DISCLAIMER,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.condition = condition
    app.state.new_guard = new_guard
    log.warning("psychosis-guard: %s", DISCLAIMER)
    log.info(
        "condition=%s chat=%s judge=%s rewriter=%s",
        condition, settings.chat, settings.judge, settings.rewriter,
    )

    # ── auth ────────────────────────────────────────────────────────────
    def require_auth(authorization: str | None = Header(default=None)) -> None:
        token = settings.auth_token
        if not token:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    auth = [Depends(require_auth)]

    # ── helpers ─────────────────────────────────────────────────────────
    def result_for(session: Session, record: TurnRecord) -> GuardResult:
        g = session.guard
        return GuardResult(
            session_id=session.id,
            reply=record.final_reply,
            intervened=record.final_reply != record.raw_reply,
            level=record.level.name,
            risk=round(record.composite_risk, 4),
            pre_level=_pre_level_from_trace(record),
            signals={k: round(v, 4) for k, v in record.signals.as_dict().items()},
            trajectory=g.summary(),
            trace=record.stage_trace if settings.include_trace else None,
        )

    def run_turn(session: Session, message: str, bot_reply: str | None = None) -> TurnRecord:
        with session.lock:
            try:
                session.guard.send(message, bot_reply=bot_reply)
            except Exception as e:
                log.exception("turn failed for session %s", session.id)
                raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e
            return session.guard.log[-1]

    # ── routes ──────────────────────────────────────────────────────────
    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "condition": condition,
            "chat": list(settings.chat),
            "judge": list(settings.judge),
            "rewriter": list(settings.rewriter),
            "sessions": len(store),
            "disclaimer": DISCLAIMER,
        }

    @app.post("/v1/guard/turn", response_model=GuardResult, dependencies=auth)
    def guard_turn(req: TurnRequest) -> GuardResult:
        sid = req.session_id or uuid.uuid4().hex
        session = store.get_or_create(sid, new_guard)
        record = run_turn(session, req.message)
        return result_for(session, record)

    @app.post("/v1/guard/check", response_model=GuardResult, dependencies=auth)
    def guard_check(req: CheckRequest) -> GuardResult:
        sid = req.session_id or uuid.uuid4().hex
        session = store.get_or_create(sid, new_guard)
        record = run_turn(session, req.user_message, bot_reply=req.bot_reply)
        return result_for(session, record)

    @app.get("/v1/sessions/{session_id}", dependencies=auth)
    def get_session(session_id: str) -> dict[str, Any]:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return {"session_id": session.id, **session.guard.summary()}

    @app.delete("/v1/sessions/{session_id}", dependencies=auth, status_code=204)
    def delete_session(session_id: str) -> Response:
        if not store.delete(session_id):
            raise HTTPException(status_code=404, detail="unknown session")
        return Response(status_code=204)

    @app.post("/v1/chat/completions", dependencies=auth)
    def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
        x_session_id: str | None = Header(default=None),
    ) -> JSONResponse:
        turns = [Turn(_role(m.role), _content_text(m.content)) for m in req.messages
                 if m.role in ("user", "assistant")]
        if not turns or turns[-1].role != Role.USER:
            raise HTTPException(status_code=400, detail="last message must be from the user")
        user_text = turns[-1].text
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="empty user message")

        sid = x_session_id or req.user
        if sid:
            session = store.get_or_create(sid, new_guard)
            stateless = False
        else:
            # stateless: rebuild the trajectory from the client-sent history
            guard = new_guard()
            guard.prime(turns[:-1])
            session = Session(id="stateless", guard=guard, created_at=time.time(),
                              last_seen=time.time())
            stateless = True

        record = run_turn(session, user_text)
        result = result_for(session, record)
        body = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "psychosis-guard",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": record.final_reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "psychosis_guard": {
                **result.model_dump(exclude={"reply"}),
                "stateless": stateless,
            },
        }
        headers = {
            "X-Guard-Level": record.level.name,
            "X-Guard-Risk": f"{record.composite_risk:.3f}",
            "X-Guard-Intervened": "1" if result.intervened else "0",
        }
        if not stateless:
            headers["X-Session-Id"] = session.id
        return JSONResponse(body, headers=headers)

    return app


# ── small helpers ───────────────────────────────────────────────────────────
def _role(r: str) -> Role:
    return Role.USER if r == "user" else Role.BOT


def _content_text(content: Any) -> str:
    """OpenAI `content` may be a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


def _pre_level_from_trace(record: TurnRecord) -> str | None:
    for line in record.stage_trace:
        if line.startswith("[stage1:input]") and "pre_level=" in line:
            return line.split("pre_level=", 1)[1].split()[0]
    return None


# module-level app for `uvicorn psychosis_guard.server.app:app`
def __getattr__(name: str) -> Any:
    if name == "app":
        return create_app()
    raise AttributeError(name)
