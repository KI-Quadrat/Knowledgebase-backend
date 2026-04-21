# ── Stage 1: dependency builder ───────────────────────────────────────────────
# We install all Python deps here, then copy only what's needed into the final
# image. This keeps build tools (uv, gcc headers) out of the runtime layer.
FROM python:3.13-slim AS builder

# Build-time labels (injected by CI/CD via --build-arg)
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
ARG VERSION=1.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    # lxml needs these at compile time (some platforms build from source)
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Pin uv to a specific version for reproducible CI builds.
# Update this intentionally when you want to upgrade.
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /bin/

WORKDIR /app

# Copy only the dependency manifests first.
# Docker caches this layer — deps only reinstall when pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment inside /app/.venv
# --no-dev:             skip dev/test packages
# --frozen:             respect uv.lock exactly, no resolution
# --no-install-project: don't install the app itself yet (just deps)
RUN uv sync --no-dev --frozen --no-install-project

# Now copy the app source and install the project itself
COPY README.md .
COPY app/ ./app/

RUN uv sync --no-dev --frozen


# ── Stage 2: runtime image ────────────────────────────────────────────────────
# Lean, no build tools. Only the venv + app code comes in.
FROM python:3.13-slim AS runtime

# OCI standard image labels — visible in your registry UI and `docker inspect`
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
ARG VERSION=1.0.0

LABEL org.opencontainers.image.title="ki2-data-plane" \
      org.opencontainers.image.description="KI² Data Plane — unified ingestion, embedding, and search for municipality RAG pipelines" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/your-org/ki2-data-plane"

# Runtime-only system dependencies (no *-dev headers needed here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    # lxml runtime
    libxml2 \
    libxslt1.1 \
    # smbprotocol — Kerberos runtime libs for AD/Windows file share auth
    libkrb5-3 \
    libgssapi-krb5-2 \
    && rm -rf /var/lib/apt/lists/*

# Copy pinned uv binary from builder (needed to run `uv run uvicorn`)
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /bin/uv

WORKDIR /app

# Copy the fully-resolved virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app   /app/app
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Non-root user — never run application code as root in production
RUN groupadd -r dataplane && useradd -r -u 1000 -g dataplane dataplane \
    && mkdir -p /tmp/uv-cache \
    && chown -R dataplane:dataplane /app /tmp/uv-cache

# Runtime environment — all secrets/config come from the environment at deploy time,
# never baked into the image. See .env.example for the full list.
ENV UV_CACHE_DIR=/tmp/uv-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

USER 1000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
