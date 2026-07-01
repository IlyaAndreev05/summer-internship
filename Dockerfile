FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.13-slim

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
RUN mkdir -p data docs projects tests

CMD ["uv", "run", "alina-rag", "console"]
