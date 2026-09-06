"""Talk to a running psychosis-guard server through the plain OpenAI SDK.

    psychosis-guard serve                     # in another terminal
    python examples/client_openai_sdk.py

Nothing in the client changes except `base_url`; the middleware wraps the
upstream chatbot and adds risk headers + a `psychosis_guard` field.
"""
from __future__ import annotations

import os
import uuid

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("PG_URL", "http://127.0.0.1:8080/v1"),
    api_key=os.environ.get("PG_AUTH_TOKEN", "not-needed"),  # sent as the bearer token
)
session_id = uuid.uuid4().hex
history: list[dict[str, str]] = []

for text in [
    "Lately I keep noticing patterns everywhere. Maybe the universe is sending me signals?",
    "I'm certain now. The signs are for me. My family don't believe me but you understand.",
]:
    history.append({"role": "user", "content": text})
    raw = client.chat.completions.with_raw_response.create(
        model="guarded",
        messages=history,
        extra_headers={"X-Session-Id": session_id},
    )
    completion = raw.parse()
    reply = completion.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    print(f"you> {text}\nbot> {reply}")
    print(f"     level={raw.headers.get('X-Guard-Level')} risk={raw.headers.get('X-Guard-Risk')}\n")
