FROM python:3.14.6-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src/ ./src/
RUN mkdir -p data docs projects tests

CMD ["sleep", "infinity"]
