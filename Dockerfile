# psychosis-guard — trajectory-aware safety middleware (HTTP)
# Build:  docker build -t psychosis-guard .
# Run:    docker run --rm -p 8080:8080 --env-file .env psychosis-guard
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# install the package with server + both provider extras
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install ".[server,openai,anthropic]"

# default policy config (override by mounting a file and setting PG_CONFIG)
COPY config.yml /app/config.yml
ENV PG_CONFIG=/app/config.yml \
    PG_HOST=0.0.0.0 \
    PG_PORT=8080

# run as non-root
RUN useradd --create-home --uid 10001 guard
USER guard

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status==200 else 1)"

CMD ["psychosis-guard", "serve"]
