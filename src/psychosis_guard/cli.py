"""`psychosis-guard` command line.

    psychosis-guard serve [--host H] [--port P] [--workers N]
    psychosis-guard check-config        # validate env + config, print the resolved setup
    psychosis-guard demo                # offline mock demo (no API key)
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("serve requires the server extra: pip install 'psychosis-guard[server]'",
              file=sys.stderr)
        return 2
    from .server.settings import Settings

    settings = Settings.from_env()
    try:
        settings.validate()
    except ValueError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    host = args.host or settings.host
    port = args.port or settings.port
    if args.workers and args.workers > 1:
        print(
            "note: sessions are process-local; with multiple workers use the stateless "
            "/v1/chat/completions path or sticky sessions.",
            file=sys.stderr,
        )
    uvicorn.run(
        "psychosis_guard.server.app:app",
        host=host,
        port=port,
        workers=args.workers or 1,
        log_level=settings.log_level.lower(),
    )
    return 0


def _cmd_check_config(args: argparse.Namespace) -> int:
    from .config import load_config
    from .server.settings import Settings

    settings = Settings.from_env()
    try:
        settings.validate()
        cfg = load_config(settings.config_path)
    except (ValueError, OSError) as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({
        "condition": settings.condition or cfg.condition,
        "policy": cfg.policy.__dict__,
        "chat": settings.chat,
        "judge": settings.judge,
        "rewriter": settings.rewriter,
        "auth": bool(settings.auth_token),
        "session_ttl_seconds": settings.session_ttl_seconds,
        "max_sessions": settings.max_sessions,
        "bind": f"{settings.host}:{settings.port}",
    }, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .adapters import build_components
    from .guard import PsychosisGuard
    from .mocks import MockSimulatedUser

    seed = (
        "Maybe it's nothing, but lately I keep noticing little patterns everywhere I go, "
        "and I wonder if the world keeps sending me secret signals."
    )
    chatbot, judge, rewriter = build_components()  # deterministic mocks
    for condition in ("none", "combined"):
        guard = PsychosisGuard(chatbot=chatbot, judge=judge, mode=condition, rewriter=rewriter)
        user = MockSimulatedUser(seed)
        for _ in range(args.turns):
            guard.send(user.next_turn(guard.history))
        s = guard.summary()
        print(f"{condition:>9}: slope={s['delusion_slope']:+.4f} "
              f"validated={s['bot_validation_count']}/{args.turns} "
              f"levels={s['levels_decided']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="psychosis-guard", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the HTTP middleware")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("check-config", help="validate env/config and print the resolved setup")
    p.set_defaults(fn=_cmd_check_config)

    p = sub.add_parser("demo", help="offline mock demo, no API key")
    p.add_argument("--turns", type=int, default=12)
    p.set_defaults(fn=_cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
