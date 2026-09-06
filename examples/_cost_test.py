"""One-off cost probe: runs a couple of turns through the real OpenAI-backed
guard and reports token usage + an estimated cost for gpt-4o-mini.

    python examples/_cost_test.py
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from psychosis_guard import PsychosisGuard
from psychosis_guard.adapters.llm import LLMChatbot, LLMJudge, LLMRewriter
from psychosis_guard.adapters.openai_adapter import OpenAICompleter

# gpt-4o-mini list pricing per 1M tokens (check platform.openai.com/pricing for current rates)
PRICE_IN = 0.150
PRICE_OUT = 0.600

usage_log: list[tuple[int, int]] = []


class TrackedOpenAICompleter(OpenAICompleter):
    def complete(self, system, messages, *, max_tokens, temperature):
        payload = ([{"role": "system", "content": system}] if system else []) + messages
        resp = self.client.chat.completions.create(
            model=self.model, messages=payload, max_tokens=max_tokens, temperature=temperature
        )
        if resp.usage:
            usage_log.append((resp.usage.prompt_tokens, resp.usage.completion_tokens))
        choice = resp.choices[0] if resp.choices else None
        return (choice.message.content if choice and choice.message else None) or ""


def main() -> None:
    client = OpenAI()
    llm = TrackedOpenAICompleter("gpt-4o-mini", client=client)
    guard = PsychosisGuard(
        chatbot=LLMChatbot(llm, base_system_prompt="You are a friendly assistant."),
        judge=LLMJudge(llm),
        rewriter=LLMRewriter(llm),
        mode="combined",
    )

    test_messages = [
        "Hi, how's the weather today?",
        "I've been getting messages from the TV that only I can decode. It's confirming I'm the chosen one.",
    ]

    for msg in test_messages:
        reply = guard.send(msg)
        rec = guard.log[-1]
        print(f"\nyou> {msg}")
        print(f"bot> {reply}")
        print(f"     [level={rec.level.name} risk={rec.composite_risk:.2f}]")

    total_in = sum(i for i, _ in usage_log)
    total_out = sum(o for _, o in usage_log)
    cost = total_in / 1_000_000 * PRICE_IN + total_out / 1_000_000 * PRICE_OUT

    print(f"\n--- {len(usage_log)} API calls across {len(test_messages)} guard.send() turns ---")
    print(f"input tokens:  {total_in}")
    print(f"output tokens: {total_out}")
    print(f"estimated cost: ${cost:.6f}  (gpt-4o-mini @ ${PRICE_IN}/1M in, ${PRICE_OUT}/1M out)")


if __name__ == "__main__":
    main()
