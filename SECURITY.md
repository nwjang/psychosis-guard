# Security policy

## Supported versions

The latest minor release on the `main` branch receives fixes.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's private
vulnerability reporting on this repository, or contact the maintainers directly.
Include a minimal reproduction and the affected version. You will get an
acknowledgement within a few days.

## Deployment notes

- Set `PG_AUTH_TOKEN` on any server reachable beyond localhost; `/healthz` stays
  public by design and does not leak credentials.
- The server never logs message contents at `INFO`; enable `DEBUG` only in trusted
  environments.
- Conversation state lives in process memory only and expires with
  `PG_SESSION_TTL_SECONDS`. Nothing is persisted to disk by the middleware.
- Provider credentials are read from environment variables and are never returned
  by any endpoint.
