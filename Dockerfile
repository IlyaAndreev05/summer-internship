FROM python:3.14.6-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev && \
    ln -s /app/.venv/bin/alina-rag /usr/local/bin/gpss-helper

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh && mkdir -p data docs projects tests

ENTRYPOINT ["./docker-entrypoint.sh"]
