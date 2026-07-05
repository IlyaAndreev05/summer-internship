# ALINA GPSS AI Consultant

RAG-консультант по системе имитационного моделирования ALINA GPSS.

## Быстрый старт (Docker)

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f ollama    # ждать загрузки моделей (~2 мин)
docker compose exec -it app gpss-helper console
```

Индексация документов из `data/docs/` и `data/projects/` происходит автоматически при запуске.

## Режимы

```bash
gpss-helper console   # интерактивный чат
gpss-helper batch     # пакетная обработка data/questions/input → data/questions/output
gpss-helper test      # тестирование с LLM-судьёй (1-10)
gpss-helper vk        # VK-бот (нужны VK_TOKEN, VK_GROUP_ID)
```

Или через `APP_MODE=console|batch|test|vk` в `.env`.

В консоли: `/exit`, `/clear`, `/verbose`.

## Структура данных

```
data/
├── docs/                  # документы: PDF, DOCX, TXT, MD
├── projects/              # таблицы проектов: XLSX, CSV (колонки: name, description)
└── questions/
    ├── input/             # вопросы для batch/test режимов
    └── output/            # результаты
```

## Варианты моделей

### LLM (генерация ответов)

| Уровень | Модель | ОЗУ | Скорость |
|---------|--------|-----|----------|
| Быстрая | `qwen2.5:0.5b` | ~0.5 GB | ★★★★★ |
| **По умолчанию** | **`qwen2.5:1.5b`** | ~1.5 GB | ★★★★ |
| Сильная | `qwen2.5:3b` | ~3 GB | ★★★ |
| Максимум | `qwen2.5:7b` | ~7 GB | ★★ |

### Embeddings (поиск)

| Уровень | Модель | Размерность | Языки |
|---------|--------|-------------|-------|
| Быстрая | `paraphrase-multilingual:latest` | 384 | 50+ |
| **По умолчанию** | **`nomic-embed-text`** | 768 | multi |
| Сильная | `bge-m3` | 1024 | 100+ |

Настройка в `.env`: `LLM_MODEL=...`, `EMBED_MODEL=...`.

## Переменные окружения

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `APP_MODE` | `console` | Режим запуска |
| `LLM_MODEL` | `qwen2.5:1.5b` | Модель LLM |
| `LLM_BASE_URL` | `http://ollama:11434` | URL Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Модель эмбеддингов |
| `QDRANT_URL` | `http://qdrant:6333` | URL Qdrant |
| `QDRANT_COLLECTION` | `alina_docs` | Коллекция Qdrant |
| `POSTGRES_URL` | `postgresql://alina:alina@postgres:5432/alina` | PostgreSQL |
| `VK_TOKEN` | — | Токен VK |
| `VK_GROUP_ID` | — | ID группы VK |
| `DATA_DIR` | `data` | Папка данных |
| `CHUNK_SIZE` | `500` | Размер чанка |
| `CHUNK_OVERLAP` | `100` | Перекрытие чанков |
| `CHAT_VERBOSE` | `false` | Отладка поиска |

## Docker-сервисы

| Сервис | Порт | Память |
|--------|------|--------|
| `app` | — | 2 GB |
| `ollama` | 11434 | 8 GB |
| `qdrant` | 6333 | 1 GB |
| `postgres` | 5432 | 512 MB |

## Локальная разработка

```bash
cp .env.example .env
# заменить адреса на localhost: LLM_BASE_URL, QDRANT_URL, POSTGRES_URL
uv sync
gpss-helper console
```

Требования: Python 3.12+, Ollama, Qdrant, PostgreSQL.
