# Architecture

psychosis-guard wraps any chatbot behind a `Chatbot.respond()` interface and runs
every user turn through five rail stages. The design follows the staged-rails
pattern of NVIDIA NeMo Guardrails (input / dialog / output / action) and adds one
stage that has no NeMo analogue: the **Trajectory Rail**.

![pipeline](assets/architecture.png)

## Pipeline

```
user turn
  │
  ▼
STAGE 1  INPUT RAIL        lexical user-side signals + PRIOR slope  → pre_level
  │        pre_level == HIGH and mode intervenes → SHORT-CIRCUIT (safe reply, chatbot never called)
  ▼
STAGE 2  DIALOG RAIL       Mode A: system_prompt_for(pre_level) → chatbot.respond(history, prompt)
  │        (or: externally supplied bot_reply — check-only path; Mode A n/a)
  ▼
STAGE 3  OUTPUT RAIL       RiskSignalDetector: lexical + judge signals on (user_text, raw_reply)
  ▼
STAGE 4  TRAJECTORY RAIL   tracker.update(...); slope = delusion_slope(window)   ★
  │        (single-turn-filter: state still updated, slope withheld → 0.0)
  ▼
STAGE 5  ACTION RAIL       risk = composite_risk(signals, slope); level = decide(risk)
                           final_reply = executor.post_process(...)  (Mode B)
```

Each stage appends a line to `TurnRecord.stage_trace`, so a turn is fully
inspectable after the fact (`trace` in HTTP responses).

## Components (`src/psychosis_guard/`)

| Module | Role |
|---|---|
| `guard.py` | `PsychosisGuard` orchestrator: `send()`, `prime()`, `summary()` |
| `rails/` | one class per stage |
| `detector.py` | `RiskSignalDetector`: fuses API-free lexicon features with judge scores |
| `lexicons.py` | regex lexicons for user-side delusion / conviction / isolation and bot-side validation / pushback |
| `tracker.py` | `TrajectoryState` time series and `delusion_slope()` (least squares) |
| `policy.py` | `composite_risk()` and threshold `decide()` |
| `executor.py` | Mode A prompts, Mode B rewrite / deterministic append |
| `config.py` | `PolicyConfig`, `GuardConfig`, strict YAML loader |
| `interfaces.py` | `Chatbot`, `Judge`, `Rewriter`, `SimulatedUser` Protocols |
| `mocks.py` | deterministic implementations of every interface (no API key) |
| `adapters/` | real-LLM implementations: provider-neutral `LLMChatbot/LLMJudge/LLMRewriter` over a `TextCompleter`; OpenAI-compatible and Anthropic completers; `build_components()` factory |
| `server/` | FastAPI app, env settings, TTL/LRU session store |
| `cli.py` | `psychosis-guard serve / check-config / demo` |

## Signals

Per turn, `Signals` holds nine floats in [0, 1].

User-side (scored on the user message): `delusion_density`, `conviction`,
`isolation`. Always computed from the lexicons; a real-LLM judge may return
language-independent estimates too, in which case the detector takes the max.

Bot-side (scored on the reply): `reinforcement`, `sycophancy`, `pushback`,
`escalation`, `help_referral` from the judge, plus the lexical summary
`bot_validated` (validation hits outnumber pushback hits).

## Risk

```
base  = w_reinforcement·reinforcement + w_sycophancy·sycophancy + w_delusion·delusion_density
      + w_conviction·conviction + w_isolation·isolation
risk  = min(base + slope_boost · max(slope, 0), 1)
level = HIGH if risk ≥ high, MEDIUM if ≥ medium, LOW if ≥ low, else NONE
```

The policy reads the slope over the last `slope_window` turns so it reacts to the
current trend; the whole-conversation slope is reported in `summary()`.

## Interventions

| Level | Mode A (system prompt before the reply) | Mode B (after the reply) |
|---|---|---|
| LOW | add a gentle grounding / reality-checking question | append or rewrite with one grounding question |
| MEDIUM | do not validate; offer a factual alternative; suggest trusted people | rewrite removing validation, add alternative explanation |
| HIGH | de-escalate; no role-play of the belief; express concern; refer to professional / crisis resources | rewrite for de-escalation and referral |

A HIGH pre-level at Stage 1 blocks the chatbot call entirely and returns a fixed
safe reply. All intervention text is marked `ADVISER-REVIEW`.

## Modes

`PsychosisGuard(mode=...)` / `condition:` in YAML:

- `none`, `detect-only` — never touch the conversation (observe / shadow).
- `A`, `B`, `combined` — intervene with the Trajectory Rail on.
- `single-turn-filter` — A + B with the slope withheld from the policy. This is
  the turn-local baseline; the only difference from `combined` is Stage 4.

## Deployment paths

- **Session-keyed**: one `PsychosisGuard` per conversation in a TTL/LRU store; the
  judge scores every turn.
- **Stateless**: `guard.prime(history)` rebuilds the trajectory from a
  client-supplied message list using the lexicons only (no judge calls), then the
  live turn runs normally. Used by `/v1/chat/completions` when no session id is
  given.
- **Check-only**: `guard.send(user_text, bot_reply=...)` scores and post-processes
  a reply produced elsewhere. Mode A cannot apply; Mode B and the short-circuit can.

## Limitations

- The lexicons are English-only; Stage 1 and stateless priming rely on them.
- The judge is an LLM rating; its agreement with human raters must be validated
  for your deployment before the risk levels are relied upon.
- Not a clinical instrument. See the disclaimer in the README.
