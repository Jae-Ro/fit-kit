# Build stage
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project --extra core --extra backend

COPY src/ /app/src/
RUN uv sync --frozen --no-dev --extra core --extra backend

# pre-download FashionSigLIP + tokenizer so they're cached in the image
RUN /app/.venv/bin/python -c "\
import open_clip; \
open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP'); \
open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP'); \
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('timm/ViT-B-16-SigLIP'); \
print('model + tokenizer cached')"

# Runtime stage
FROM python:3.12-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

CMD ["taskiq", "worker", "serve.worker:broker"]
