"""HTTP deployment of psychosis-guard (extra: `pip install psychosis-guard[server]`).

    psychosis-guard serve            # reads PG_* / provider env vars
    uvicorn psychosis_guard.server.app:app

Endpoints (see app.py): /healthz, /v1/guard/turn, /v1/guard/check,
/v1/chat/completions (OpenAI-compatible proxy), /v1/sessions/{id}.
"""
