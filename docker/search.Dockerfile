# Build stage
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project --extra core --extra search

COPY src/ /app/src/
RUN uv sync --frozen --no-dev --extra core --extra search

# Runtime stage
FROM python:3.12-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

EXPOSE 50051
CMD ["python", "-m", "src.search_service.server"]
