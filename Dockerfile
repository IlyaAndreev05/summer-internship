FROM python:3.14-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

COPY src/ src/
RUN uv sync --no-dev

RUN printf '#!/bin/sh\nexec /app/.venv/bin/gpss-helper "$@"\n' > /usr/local/bin/gpss-helper && chmod +x /usr/local/bin/gpss-helper

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
