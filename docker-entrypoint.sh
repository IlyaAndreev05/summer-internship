#!/bin/sh
set -e

MODE="${APP_MODE:-console}"
PYTHON="/app/.venv/bin/python"

echo "=== ALINA RAG | mode=$MODE ==="

case "$MODE" in
  index)
    echo "Indexing documents..."
    $PYTHON -c "from alina_rag.auto_index import auto_index; auto_index()"
    echo "Done."
    ;;
  console|vk|test|batch)
    exec $PYTHON -m alina_rag.main
    ;;
  *)
    echo "Unknown APP_MODE: $MODE. Use: console, vk, test, batch, index"
    exit 1
    ;;
esac
