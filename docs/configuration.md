# Configuration

## Policy YAML (`config.yml`, `PG_CONFIG`)

```yaml
condition: combined        # none | detect-only | single-turn-filter | A | B | combined
policy:
  low: 0.25                # thresholds on composite risk
  medium: 0.50
  high: 0.75
  w_reinforcement: 0.35    # signal weights
  w_sycophancy: 0.20
  w_delusion: 0.20
  w_conviction: 0.15
  w_isolation: 0.10
  slope_boost: 3.0         # gain on max(slope, 0)
  slope_window: 4          # turns the policy looks back over (null = whole conversation)
```

Unknown keys are rejected at load time. Presets for each mode are in `configs/`.

Tuning notes: densities live in [0, 1], so a slope of ~0.1/turn is already a steep
climb; with `slope_boost: 3.0` that adds +0.3 to the risk. Lower `low/medium/high`
for a more sensitive deployment, or raise `slope_boost` to make the middleware
react more to drift than to single turns.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `PG_CONFIG` | path to the policy YAML | package defaults |
| `PG_CONDITION` | override the YAML's `condition` | — |
| `PG_CHAT_PROVIDER` | `mock` \| `openai` \| `anthropic` | `mock` |
| `PG_CHAT_MODEL` | chatbot model (required for `openai`) | Anthropic: `claude-opus-5` |
| `PG_CHAT_SYSTEM_PROMPT` | your application's own system prompt; Mode A guidance is appended | — |
| `PG_JUDGE_PROVIDER` / `PG_JUDGE_MODEL` | judge model | same as chat |
| `PG_JUDGE_EFFORT` | Anthropic `effort` for judge/rewriter calls | `low` |
| `PG_REWRITER_PROVIDER` / `PG_REWRITER_MODEL` | rewriter model; `none` = deterministic append | same as judge |
| `OPENAI_API_KEY` | OpenAI credential | — |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint (Ollama, vLLM, …) | — |
| `ANTHROPIC_API_KEY` | Anthropic credential | — |
| `PG_AUTH_TOKEN` | bearer token required on `/v1/*` | off |
| `PG_SESSION_TTL_SECONDS` | idle time before a session is dropped | `3600` |
| `PG_MAX_SESSIONS` | LRU cap on stored sessions | `10000` |
| `PG_TRACE` | include per-stage trace in responses | `true` |
| `PG_HOST` / `PG_PORT` | bind address for `psychosis-guard serve` | `0.0.0.0` / `8080` |
| `PG_LOG_LEVEL` | Python logging level | `INFO` |

`psychosis-guard check-config` prints the resolved setup and exits non-zero on an
invalid combination.

## Library construction

```python
from psychosis_guard import PsychosisGuard, PolicyConfig
from psychosis_guard.adapters import build_components

chatbot, judge, rewriter = build_components(
    chat=("openai", "gpt-4o-mini"),
    judge=("anthropic", "claude-opus-5"),
    rewriter=("none", None),
    chat_system_prompt="You are a friendly assistant.",
)
guard = PsychosisGuard(chatbot, judge, mode="combined", rewriter=rewriter,
                       policy=PolicyConfig(slope_boost=4.0))
```
