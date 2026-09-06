"""Real-LLM adapters and the component factory used by the server/CLI.

    from psychosis_guard.adapters import build_components
    chatbot, judge, rewriter = build_components(chat=("openai", "gpt-4o-mini"))

Providers: "mock" (no API key), "openai" (any OpenAI-compatible base_url),
"anthropic". Judge and rewriter can use a different, cheaper provider/model
than the chatbot under guard.
"""
from __future__ import annotations

from ..interfaces import Chatbot, Judge, Rewriter
from .llm import LLMChatbot, LLMJudge, LLMRewriter, TextCompleter

PROVIDERS = ("mock", "openai", "anthropic")


def make_completer(
    provider: str,
    model: str | None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    effort: str | None = None,
) -> TextCompleter:
    provider = provider.lower()
    if provider == "openai":
        if not model:
            raise ValueError("openai provider needs a model name")
        from .openai_adapter import OpenAICompleter

        return OpenAICompleter(model, api_key=api_key, base_url=base_url)
    if provider == "anthropic":
        from .anthropic_adapter import DEFAULT_MODEL, AnthropicCompleter

        return AnthropicCompleter(model or DEFAULT_MODEL, api_key=api_key, effort=effort)
    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def build_components(
    *,
    chat: tuple[str, str | None] = ("mock", None),
    judge: tuple[str, str | None] | None = None,
    rewriter: tuple[str, str | None] | None = None,
    chat_system_prompt: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    anthropic_api_key: str | None = None,
    judge_effort: str | None = "low",
) -> tuple[Chatbot, Judge, Rewriter | None]:
    """Build (chatbot, judge, rewriter) from (provider, model) pairs.

    `judge` defaults to the chat pair; `rewriter` defaults to the judge pair.
    Provider "none" for the rewriter selects the deterministic Mode-B append.
    """
    from ..mocks import MockChatbot, MockJudge, MockRewriter

    def _completer(pair: tuple[str, str | None], effort: str | None = None) -> TextCompleter:
        prov, model = pair
        key = openai_api_key if prov == "openai" else anthropic_api_key
        return make_completer(prov, model, api_key=key, base_url=openai_base_url, effort=effort)

    judge = judge or chat
    rewriter = rewriter or judge

    chatbot: Chatbot = (
        MockChatbot()
        if chat[0] == "mock"
        else LLMChatbot(_completer(chat), base_system_prompt=chat_system_prompt)
    )
    judge_obj: Judge = (
        MockJudge() if judge[0] == "mock" else LLMJudge(_completer(judge, judge_effort))
    )
    rewriter_obj: Rewriter | None
    if rewriter[0] == "none":
        rewriter_obj = None
    elif rewriter[0] == "mock":
        rewriter_obj = MockRewriter()
    else:
        rewriter_obj = LLMRewriter(_completer(rewriter, judge_effort))
    return chatbot, judge_obj, rewriter_obj


__all__ = [
    "PROVIDERS",
    "LLMChatbot",
    "LLMJudge",
    "LLMRewriter",
    "TextCompleter",
    "build_components",
    "make_completer",
]
