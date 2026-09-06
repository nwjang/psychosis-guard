"""psychosis-guard with a real chatbot (library use, no server).

    export OPENAI_API_KEY=...            # or OPENAI_BASE_URL for Ollama/vLLM
    python examples/quickstart_openai.py --model gpt-4o-mini

Wraps an OpenAI-compatible model as the chatbot under guard and uses the same
model as judge + rewriter. Swap in `AnthropicCompleter` for Claude.
"""
from __future__ import annotations

import argparse
import os

from psychosis_guard import PsychosisGuard
from psychosis_guard.adapters.llm import LLMChatbot, LLMJudge, LLMRewriter
from psychosis_guard.adapters.openai_adapter import OpenAICompleter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("PG_CHAT_MODEL", "gpt-4o-mini"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    args = ap.parse_args()

    llm = OpenAICompleter(args.model, base_url=args.base_url)
    guard = PsychosisGuard(
        chatbot=LLMChatbot(llm, base_system_prompt="You are a friendly assistant."),
        judge=LLMJudge(llm),
        rewriter=LLMRewriter(llm),
        mode="combined",
    )

    print("type a message (Ctrl-D to quit)")
    try:
        while True:
            user = input("\nyou> ").strip()
            if not user:
                continue
            reply = guard.send(user)
            rec = guard.log[-1]
            print(f"bot> {reply}")
            print(f"     [level={rec.level.name} risk={rec.composite_risk:.2f} "
                  f"slope={guard.tracker.state.delusion_slope():+.3f}]")
    except EOFError:
        print()


if __name__ == "__main__":
    main()
