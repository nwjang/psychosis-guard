# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- README **Evaluation** section: psychosis-bench (arXiv:2509.10970) results for
  the unguarded chatbot, a safety system prompt, and the middleware modes,
  with the Trajectory Rail ablation and the reactive-simulation caveat.
- Evaluation figures (`docs/eval/`) and a Korean README (`README.ko.md`).

## [0.2.0] — 2026-09-02

### Added
- HTTP middleware (`psychosis_guard.server`): OpenAI-compatible proxy
  `/v1/chat/completions`, guarded turn `/v1/guard/turn`, check-only
  `/v1/guard/check`, session endpoints, `/healthz`, bearer auth.
- Real-LLM adapters: provider-neutral `LLMChatbot` / `LLMJudge` / `LLMRewriter`
  over a `TextCompleter`; OpenAI-compatible and Anthropic backends;
  `build_components()` factory.
- Judge may return language-independent user-side signals; detector fuses them
  with the lexicons.
- `PsychosisGuard.send(..., bot_reply=)` check-only path, `prime()` stateless
  trajectory rebuild, `summary()`.
- CLI `psychosis-guard serve | check-config | demo`.
- Dockerfile, docker-compose, `.env.example`, CI Docker smoke test.
- Mode presets in `configs/`.

### Changed
- Repository reorganised as a middleware library; research harness assets moved
  out of the public tree.

## [0.1.0]

### Added
- Five-stage rails pipeline with the Trajectory Rail, config-driven policy,
  deterministic mocks, and the offline quickstart.
