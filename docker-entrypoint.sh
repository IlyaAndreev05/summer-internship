#!/bin/sh
set -e

MODE="${APP_MODE:-console}"
PYTHON="/app/.venv/bin/python"

echo "=== ALINA RAG | mode=$MODE ==="

# Авто-индексация: проверяет хеши в Postgres, индексирует только новые/изменённые
echo "Auto-indexing..."
$PYTHON -c "from alina_rag.auto_index import auto_index; auto_index()" || echo "Auto-index failed, continuing..."

case "$MODE" in
  console|vk|test|batch)
    exec $PYTHON -m alina_rag.main
    ;;
  *)
    echo "Unknown APP_MODE: $MODE. Use: console, vk, test, batch"
    exit 1
    ;;
esac
