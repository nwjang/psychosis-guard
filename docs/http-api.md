# HTTP API

Base URL: `http://<host>:8080`. Interactive docs at `/docs`. If `PG_AUTH_TOKEN` is
set, every `/v1/*` route requires `Authorization: Bearer <token>`.

## `GET /healthz`

Liveness and resolved configuration. Public.

```json
{"status":"ok","version":"0.2.0","condition":"combined","chat":["openai","gpt-4o-mini"],
 "judge":["openai","gpt-4o-mini"],"rewriter":["openai","gpt-4o-mini"],"sessions":3,"disclaimer":"..."}
```

## `POST /v1/chat/completions` — OpenAI-compatible proxy

Request: a standard Chat Completions body. `messages` (string or content-part
`content`), optional `model` (echoed back), optional `user`. Other fields are
accepted and ignored; `stream: true` is accepted but the reply is returned whole.

Session resolution: `X-Session-Id` header → body `user` → **stateless** (trajectory
rebuilt from the sent history; no session is stored).

Response: a standard chat completion plus a `psychosis_guard` object (same fields as
`GuardResult` below minus `reply`, plus `stateless`), and headers
`X-Guard-Level`, `X-Guard-Risk`, `X-Guard-Intervened`, and `X-Session-Id` when a
session was used.

Errors: `400` if the last message is not a non-empty user message; `502` if the
upstream chatbot fails.

## `POST /v1/guard/turn`

```json
{"session_id": "abc", "message": "user text"}
```

`session_id` optional; one is generated when omitted. The middleware calls the
upstream chatbot and returns a `GuardResult`.

## `POST /v1/guard/check`

```json
{"session_id": "abc", "user_message": "user text", "bot_reply": "draft reply from your bot"}
```

Scores an externally produced reply, updates the trajectory, and returns a
`GuardResult` whose `reply` is what the user should see. On a HIGH pre-level the
draft is replaced entirely.

## `GuardResult`

| Field | Type | Meaning |
|---|---|---|
| `session_id` | string | session used or created |
| `reply` | string | final reply to display |
| `intervened` | bool | `reply` differs from the raw reply |
| `level` | `NONE\|LOW\|MEDIUM\|HIGH` | decided level for this turn |
| `risk` | float | composite risk in [0, 1] |
| `pre_level` | string | Stage 1 pre-response level |
| `signals` | object | the nine per-turn signals |
| `trajectory` | object | `turn_count`, `delusion_slope`, `recent_slope`, `bot_validation_count`, `isolation_mentions`, `levels_decided`, `last_level`, `last_risk`, `mode` |
| `trace` | string[] or null | per-stage trace lines (`PG_TRACE=false` disables) |

## `GET /v1/sessions/{id}`

Trajectory summary for a session. `404` if unknown or expired.

## `DELETE /v1/sessions/{id}`

Drops the session. `204` on success, `404` if unknown.
