"""Runtime settings, read from environment variables (12-factor).

    PG_CONFIG              path to a NeMo-style config.yml (default: package defaults)
    PG_CONDITION           override the config's condition (none|detect-only|single-turn-filter|A|B|combined)
    PG_CHAT_PROVIDER       mock | openai | anthropic            (default: mock)
    PG_CHAT_MODEL          model for the chatbot under guard
    PG_CHAT_SYSTEM_PROMPT  your app's own base system prompt for the chatbot
    PG_JUDGE_PROVIDER      defaults to PG_CHAT_PROVIDER
    PG_JUDGE_MODEL         defaults to PG_CHAT_MODEL (use a cheaper model here)
    PG_JUDGE_EFFORT        Anthropic effort for judge/rewriter calls (default: low)
    PG_REWRITER_PROVIDER   defaults to judge provider; "none" = deterministic append
    PG_REWRITER_MODEL      defaults to PG_JUDGE_MODEL
    OPENAI_API_KEY / OPENAI_BASE_URL / ANTHROPIC_API_KEY   provider credentials
    PG_AUTH_TOKEN          if set, every request must send `Authorization: Bearer <token>`
    PG_SESSION_TTL_SECONDS idle time before a session is dropped (default: 3600)
    PG_MAX_SESSIONS        LRU cap on in-memory sessions (default: 10000)
    PG_TRACE               include per-stage trace lines in responses (default: true)
    PG_HOST / PG_PORT      bind address for `psychosis-guard serve` (default 0.0.0.0:8080)
    PG_LOG_LEVEL           python logging level (default: INFO)
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    config_path: str | None = None
    condition: str | None = None

    chat_provider: str = "mock"
    chat_model: str | None = None
    chat_system_prompt: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_effort: str | None = "low"
    rewriter_provider: str | None = None
    rewriter_model: str | None = None

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None

    auth_token: str | None = None
    session_ttl_seconds: int = 3600
    max_sessions: int = 10_000
    include_trace: bool = True

    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            config_path=_env("PG_CONFIG"),
            condition=_env("PG_CONDITION"),
            chat_provider=(_env("PG_CHAT_PROVIDER", "mock") or "mock").lower(),
            chat_model=_env("PG_CHAT_MODEL"),
            chat_system_prompt=_env("PG_CHAT_SYSTEM_PROMPT"),
            judge_provider=(_env("PG_JUDGE_PROVIDER") or "").lower() or None,
            judge_model=_env("PG_JUDGE_MODEL"),
            judge_effort=_env("PG_JUDGE_EFFORT", "low"),
            rewriter_provider=(_env("PG_REWRITER_PROVIDER") or "").lower() or None,
            rewriter_model=_env("PG_REWRITER_MODEL"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_base_url=_env("OPENAI_BASE_URL"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            auth_token=_env("PG_AUTH_TOKEN"),
            session_ttl_seconds=int(_env("PG_SESSION_TTL_SECONDS", "3600") or 3600),
            max_sessions=int(_env("PG_MAX_SESSIONS", "10000") or 10000),
            include_trace=_env_bool("PG_TRACE", True),
            host=_env("PG_HOST", "0.0.0.0") or "0.0.0.0",
            port=int(_env("PG_PORT", "8080") or 8080),
            log_level=(_env("PG_LOG_LEVEL", "INFO") or "INFO").upper(),
        )

    # resolved (provider, model) pairs -------------------------------------
    @property
    def chat(self) -> tuple[str, str | None]:
        return (self.chat_provider, self.chat_model)

    @property
    def judge(self) -> tuple[str, str | None]:
        prov = self.judge_provider or self.chat_provider
        model = self.judge_model or (self.chat_model if prov == self.chat_provider else None)
        return (prov, model)

    @property
    def rewriter(self) -> tuple[str, str | None]:
        jprov, jmodel = self.judge
        prov = self.rewriter_provider or jprov
        model = self.rewriter_model or (jmodel if prov == jprov else None)
        return (prov, model)

    def validate(self) -> None:
        from ..adapters import PROVIDERS
        from ..executor import VALID_MODES

        for label, (prov, model) in (
            ("chat", self.chat),
            ("judge", self.judge),
            ("rewriter", self.rewriter),
        ):
            if label == "rewriter" and prov == "none":
                continue
            if prov not in PROVIDERS:
                raise ValueError(f"PG_{label.upper()}_PROVIDER={prov!r}; expected one of {PROVIDERS}")
            if prov == "openai" and not model:
                raise ValueError(f"PG_{label.upper()}_MODEL is required for the openai provider")
            if prov == "openai" and not (self.openai_api_key or self.openai_base_url):
                raise ValueError("OPENAI_API_KEY (or OPENAI_BASE_URL for a local server) is required")
            if prov == "anthropic" and not self.anthropic_api_key and not os.environ.get(
                "ANTHROPIC_AUTH_TOKEN"
            ):
                # the SDK can also resolve an `ant auth login` profile; warn only
                pass
        if self.condition is not None and self.condition not in VALID_MODES:
            raise ValueError(
                f"PG_CONDITION={self.condition!r}; expected one of {sorted(VALID_MODES)}"
            )
        if self.session_ttl_seconds <= 0 or self.max_sessions <= 0:
            raise ValueError("PG_SESSION_TTL_SECONDS and PG_MAX_SESSIONS must be positive")
