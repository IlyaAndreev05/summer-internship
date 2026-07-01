# ALINA GPSS AI Consultant

ИИ-консультант по работе с ALINA GPSS на основе RAG (Retrieval-Augmented Generation).
LangChain + Ollama + Qdrant + PostgreSQL + BM25.

## Требования

- Docker и Docker Compose
- [uv](https://docs.astral.sh/uv/) — только для локальной разработки

## Быстрый старт (Docker Compose)

```bash
cp .env.example .env
docker compose up -d
```

Подождать пока Ollama скачает модели (около минуты):

```bash
docker compose logs -f ollama
```

Дождаться `qwen2.5:1.5b` и `nomic-embed-text`, затем `Ctrl+C`.

### Индексирование документов

Положи документацию в `docs/`, проекты в `projects/`, затем:

```bash
docker compose exec app uv run alina-rag index
```

### Консольный чат-бот

```bash
docker compose exec -it app uv run alina-rag console
```

Команды внутри чата: `/exit`, `/clear`, `/verbose`.

### Массовая обработка (batch)

Файлы из `tests/` с колонками `№, Вопрос, Ответ` (Ответ пустой — система заполнит):

```bash
docker compose exec app uv run alina-rag batch
```

Результат: `*_filled.*` в `tests/`.

### Тестирование с самооценкой (test)

Файлы из `tests/` с колонками `№, Вопрос, Ответ, Правильный ответ`. Система заполняет ответ, LLM-судья сравнивает с правильным и ставит оценку 1–10:

```bash
docker compose exec app uv run alina-rag test
```

Результат: `*_scored.*` в `tests/` + статистика в консоли.

### VK бот

Нужен `VK_TOKEN` и `VK_GROUP_ID` в `.env`:

```bash
docker compose exec app uv run alina-rag vk
```

## Локальная разработка (без Docker)

```bash
cp .env.example .env
uv sync
uv run alina-rag index
uv run alina-rag console
```

## Переменные окружения (.env)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:1.5b` | Модель Ollama |
| `LLM_BASE_URL` | `http://ollama:11434` | Адрес Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Модель для эмбеддингов |
| `QDRANT_URL` | `http://qdrant:6333` | Адрес Qdrant |
| `QDRANT_COLLECTION` | `alina_docs` | Коллекция Qdrant |
| `POSTGRES_URL` | `postgresql://alina:alina@postgres:5432/alina` | PostgreSQL |
| `VK_TOKEN` | — | Токен VK |
| `VK_GROUP_ID` | — | ID группы VK |
| `DOCS_DIR` | `docs` | Папка с документацией |
| `PROJECTS_DIR` | `projects` | Папка с проектами |
| `TESTS_DIR` | `tests` | Папка с тестами |
| `CHUNK_SIZE` | `500` | Размер чанка |
| `CHUNK_OVERLAP` | `100` | Перекрытие чанков |

## Структура папок с данными

```
docs/       — .txt .md .pdf .docx .doc
projects/   — .xlsx .xls .csv
tests/      — .csv .xlsx .xls
```

## Docker-сервисы

| Сервис | Порт | Описание |
|---|---|---|
| `app` | — | Приложение |
| `ollama` | 11434 | LLM + эмбеддинги |
| `qdrant` | 6333 | Векторная БД |
| `postgres` | 5432 | PostgreSQL 18 |

## Архитектура

```
alina_rag/
    main.py         — CLI (console, vk, index, batch, test)
    config.py       — Настройки
    prompts.py      — Системные промпты
    agent.py        — RAG: Qdrant + BM25 → LLM
    indexer.py      — Индексация документов
    console_bot.py  — Терминальный чат
    vk_bot.py       — VK бот
    batch_mode.py   — Массовая обработка
    test_mode.py    — Тест с самооценкой
```

## Инструкция администратора

### Установка

```bash
git clone <repo-url>
cd summer-internship
cp .env.example .env
```

### Настройка

1. Отредактируйте `.env` — укажите токены VK при необходимости
2. Поместите документацию в `docs/`
3. Поместите файлы проектов в `projects/`

### Запуск

```bash
docker compose up -d
docker compose logs -f ollama    # дождаться загрузки моделей
docker compose exec app uv run alina-rag index
```

### Обновление данных

```bash
docker compose exec app uv run alina-rag index
```

## Технологический стек

Python 3.14, LangChain, Ollama (qwen2.5:1.5b + nomic-embed-text), Qdrant, BM25, PostgreSQL 18, Docker Compose
