#!/bin/sh
set -e

MODE="${APP_MODE:-console}"

echo "=== ALINA RAG | mode=$MODE ==="

# Auto-index on startup (every mode)
echo "Running auto-index..."
python -c "from alina_rag.auto_index import auto_index; auto_index()" || echo "Auto-index failed, continuing..."

case "$MODE" in
  console|vk|test|batch)
    exec python -m alina_rag.main
    ;;
  *)
    echo "Unknown APP_MODE: $MODE. Use: console, vk, test, batch"
    exit 1
    ;;
esac
