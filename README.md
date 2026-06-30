# ALINA GPSS AI Consultant

RAG-консультант на базе локальных LLM для ответов на вопросы по руководству пользователя ALINA GPSS.

## Архитектура

Agentic RAG (ReAct) — агент самостоятельно решает: искать ли документы, переформулировать запрос, искать ещё, или отвечать сразу.

```
User → Agent (ReAct) → [Search Tool] → ChromaDB → LLM → Answer
                    ↘ [Direct Answer]
```

## Системные требования

- CPU: 4+ ядер
- RAM: 16 GB
- OS: Linux (основной), macOS (тестируется), Windows (через Docker)
- Python 3.11+
- [Ollama](https://ollama.com) (для локальных моделей)
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов
- [ty](https://docs.astral.sh/ty/) — проверка типов
- [ruff](https://docs.astral.sh/ruff/) — линтер и форматтер

## Сравнение моделей

### LLM (через Ollama, квантизация Q4_K_M)

| Модель | Размер на диске | RAM в работе | Контекст | Русский язык | Скорость (CPU, 4 ядра) | Качество RAG | Рекомендация |
|---|---|---|---|---|---|---|---|
| **qwen2.5:0.5b** | 398 MB | ~600 MB | 32K | ★★☆☆☆ | ~40 tok/s | ★★☆☆☆ | Запасной вариант для очень слабого железа |
| **qwen2.5:1.5b** ★ | 986 MB | ~1.4 GB | 32K | ★★★★☆ | ~25 tok/s | ★★★☆☆ | **Рекомендовано** — лучший баланс |
| **qwen2.5:3b** | 1.9 GB | ~2.8 GB | 32K | ★★★★☆ | ~12 tok/s | ★★★★☆ | Если хватает памяти |
| **gemma3:1b** | 700 MB | ~1.1 GB | 32K | ★★★☆☆ | ~30 tok/s | ★★☆☆☆ | Слабый русский |
| **gemma3:4b** | 2.5 GB | ~3.5 GB | 128K | ★★★☆☆ | ~8 tok/s | ★★★★☆ | Большой контекст, слабее русский |
| **phi4-mini:3.8b** | 2.2 GB | ~3.2 GB | 128K | ★★☆☆☆ | ~9 tok/s | ★★★★☆ | Английский, хорошая логика |

★ — рекомендовано для 16 GB RAM.

### Embedding-модели (sentence-transformers, CPU)

| Модель | Размер | RAM | Размерность | Языки | Скорость (CPU) | Качество (русский) | Рекомендация |
|---|---|---|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 80 MB | ~200 MB | 384 | EN | ~750 emb/s | ★★☆☆☆ | Очень быстрая, только EN |
| **paraphrase-multilingual-MiniLM-L12-v2** ★ | 470 MB | ~700 MB | 384 | 50+ (вкл. RU) | ~400 emb/s | ★★★★☆ | **Рекомендовано** — баланс RU/скорость |
| **intfloat/multilingual-e5-small** | 470 MB | ~700 MB | 384 | 94 (вкл. RU) | ~350 emb/s | ★★★★☆ | Чуть лучше качество, нужны префиксы |
| **distiluse-base-multilingual-cased-v2** | 540 MB | ~800 MB | 512 | 50+ (вкл. RU) | ~250 emb/s | ★★★☆☆ | Более старый, медленнее |
| **BAAI/bge-small-en-v1.5** | 130 MB | ~250 MB | 384 | EN | ~650 emb/s | ★☆☆☆☆ | Отличное качество EN, нет RU |
| **intfloat/multilingual-e5-large** | 2.2 GB | ~3 GB | 1024 | 94 | ~100 emb/s | ★★★★★ | Слишком тяжёлая для 16 GB |

★ — рекомендовано для 16 GB RAM.

### Сводка рекомендуемого стека

| Компонент | Модель | RAM |
|---|---|---|
| LLM | `qwen2.5:1.5b` (Q4_K_M) | ~1.4 GB |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | ~700 MB |
| Ollama runtime | — | ~500 MB |
| Приложение Python | — | ~500 MB |
| **Итого** | | **~3.1 GB** |

Запас ~13 GB на пиковые нагрузки, кэши и дополнительные процессы.

## Быстрый старт

### 1. Установка Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
```

### 2. Установка зависимостей

```bash
uv sync
```

### 3. Запуск

```bash
# Консольный чат
uv run alina-rag --mode console

# Telegram бот
uv run alina-rag --mode telegram

# VK бот
uv run alina-rag --mode vk

# Все режимы одновременно
uv run alina-rag --mode all

# API + вебхуки для ботов
uv run alina-rag --mode api
```

### 4. Добавление документов

```bash
# Из папки data/documents/ (автоматически при запуске)
# Или через CLI:
uv run alina-rag ingest --path ./manual.pdf
# Или через API:
curl -X POST http://localhost:8000/documents/upload -F "file=@manual.pdf"
```

## Docker

```bash
docker-compose up -d
```

## Конфигурация

Все настройки через `.env` файл или переменные окружения:

```env
# LLM
LLM_PROVIDER=ollama          # ollama | openai
LLM_MODEL=qwen2.5:1.5b       # модель для ollama
LLM_BASE_URL=http://localhost:11434  # Ollama или OpenAI-compatible URL
LLM_API_KEY=                  # для OpenAI-совместимых API

# Embeddings
EMBED_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBED_DEVICE=cpu

# SQLite
DATABASE_URL=sqlite:///data/chat_history.db

# ChromaDB
CHROMA_PERSIST_DIR=data/chroma

# Telegram
TELEGRAM_TOKEN=

# VK
VK_TOKEN=
VK_GROUP_ID=

# Chat
CHAT_MAX_MESSAGES=20          # последние N сообщений в памяти агента
```
