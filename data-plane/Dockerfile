FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

COPY . .
RUN uv sync --no-dev --frozen

RUN groupadd -r dataplane && useradd -r -g dataplane dataplane \
    && mkdir -p /tmp/uv-cache \
    && chown -R dataplane:dataplane /app /tmp/uv-cache

ENV UV_CACHE_DIR=/tmp/uv-cache
USER dataplane

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
