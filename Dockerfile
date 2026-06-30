# Stage 1: Build environment
FROM ghcr.io/astral-sh/uv:latest AS uv

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy uv binary from official image
COPY --from=uv /uv /usr/local/bin/uv

# Install project dependencies (no dev)
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# Copy application source and data directory
COPY src/ ./src/
RUN mkdir -p data/documents data/chroma

EXPOSE 8000

CMD ["uv", "run", "alina-rag", "run", "--mode", "api"]
