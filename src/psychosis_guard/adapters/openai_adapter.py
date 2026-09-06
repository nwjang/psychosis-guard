"""OpenAI Chat Completions backend (extra: `pip install psychosis-guard[openai]`).

Also covers every OpenAI-compatible server — Ollama, vLLM, LM Studio, Groq,
Together, OpenRouter, Azure-style gateways — via `base_url`.
"""
from __future__ import annotations

from typing import Any


class OpenAICompleter:
    """`TextCompleter` over `openai.OpenAI().chat.completions.create`."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAI backend requires the 'openai' package: "
                    "pip install 'psychosis-guard[openai]'"
                ) from e
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.client = client
        self.model = model
        self.extra_kwargs = extra_kwargs or {}

    def complete(
        self,
        system: str | None,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = ([{"role": "system", "content": system}] if system else []) + messages
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            max_tokens=max_tokens,
            temperature=temperature,
            **self.extra_kwargs,
        )
        choice = resp.choices[0] if resp.choices else None
        content = choice.message.content if choice and choice.message else None
        return content or ""
