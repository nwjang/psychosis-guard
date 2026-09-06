"""Anthropic Messages API backend (extra: `pip install psychosis-guard[anthropic]`)."""
from __future__ import annotations

from typing import Any

DEFAULT_MODEL = "claude-opus-5"


class AnthropicCompleter:
    """`TextCompleter` over `anthropic.Anthropic().messages.create`."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
        effort: str | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "Anthropic backend requires the 'anthropic' package: "
                    "pip install 'psychosis-guard[anthropic]'"
                ) from e
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self.client = client
        self.model = model
        self.effort = effort
        self.extra_kwargs = extra_kwargs or {}

    def complete(
        self,
        system: str | None,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,  # accepted for interface parity; current Claude models reject sampling params
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **self.extra_kwargs,
        )
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.messages.create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            # a safety refusal is not a reply to score — let the caller degrade
            return ""
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
