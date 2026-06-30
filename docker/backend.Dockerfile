# Build stage
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project --extra backend

COPY src/ /app/src/
RUN uv sync --frozen --no-dev --extra backend

# Runtime stage
FROM python:3.12-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

EXPOSE 8000
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
