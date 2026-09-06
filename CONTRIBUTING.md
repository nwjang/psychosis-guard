# Contributing to psychosis-guard

Thanks for your interest. psychosis-guard is a safety middleware for a sensitive
domain, so a few rules are stricter than usual.

## Ground rules

- **No real user data.** Never commit conversation logs, transcripts, or personas
  derived from real people. Test fixtures must be synthetic, harmless proxy content.
- **Clinical text needs review.** Any change to a line marked `ADVISER-REVIEW`
  (intervention copy in `executor.py`/`guard.py`/`mocks.py`, judge and rewriter
  prompts in `adapters/prompts.py`, the disclaimer) requires sign-off from a
  mental-health professional before merge. Say so in the PR.
- **Mock-first.** Everything must run and pass tests with no API key. Real-LLM
  behaviour goes behind the `TextCompleter` interface and is tested with fakes.
- **Clean room.** Do not copy code from NeMo Guardrails or other guardrail
  libraries; structural inspiration only.
- **No secrets.** API keys come from environment variables only.

## Development

```bash
git clone https://github.com/nwjang/psychosis-guard.git
cd psychosis-guard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check .
psychosis-guard demo
```

## Pull requests

- Small, focused PRs with conventional-commit style titles (`feat:`, `fix:`,
  `docs:`, `test:`, `refactor:`).
- Add or update tests for behaviour changes; CI must be green (lint, tests on
  Python 3.10 and 3.12, Docker build + container smoke test).
- Keep the pipeline readable: one class per rail stage, comments that explain
  *why*, not *what*.
- Update `CHANGELOG.md` under *Unreleased*.

## Reporting issues

Use GitHub Issues for bugs and feature requests. For security problems see
[SECURITY.md](SECURITY.md). Do not paste real conversations into issues.
