# ALINA GPSS AI Consultant

ИИ-консультант по работе с ALINA GPSS на основе RAG (Retrieval-Augmented Generation).
Использует LangChain, Ollama, Qdrant, PostgreSQL и BM25.

## Требования

- Python 3.12+
- Docker и Docker Compose (для production-запуска)
- [uv](https://docs.astral.sh/uv/) (для локальной разработки)

## Быстрый старт (Docker)

```bash
cp .env.example .env
docker compose up -d
docker compose exec app uv run alina-rag index
docker compose exec app uv run alina-rag console
```

## Локальная разработка

```bash
cp .env.example .env
uv sync
uv run alina-rag index
uv run alina-rag console
```

## Переменные окружения (.env)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:1.5b` | Модель Ollama для генерации |
| `LLM_BASE_URL` | `http://localhost:11434` | Адрес Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Модель Ollama для эмбеддингов |
| `QDRANT_URL` | `http://localhost:6333` | Адрес Qdrant |
| `QDRANT_COLLECTION` | `alina_docs` | Название коллекции Qdrant |
| `POSTGRES_URL` | `postgresql://alina:alina@localhost:5432/alina` | URL базы PostgreSQL |
| `VK_TOKEN` | — | Токен сообщества VK |
| `VK_GROUP_ID` | — | ID группы VK |
| `DOCS_DIR` | `docs` | Папка с документацией |
| `PROJECTS_DIR` | `projects` | Папка с проектами (xlsx/csv) |
| `TESTS_DIR` | `tests` | Папка с тестовыми файлами |
| `CHUNK_SIZE` | `500` | Размер чанка для индексации |
| `CHUNK_OVERLAP` | `100` | Перекрытие чанков |

## Команды

### `alina-rag index` — Индексирование документов

Загружает все файлы из папок `docs/` и `projects/`, разбивает на чанки и индексирует в Qdrant (векторный поиск) и BM25 (ключевые слова).

Поддерживаемые форматы: `.txt`, `.md`, `.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.csv`.

```
docs/       — документация (руководство пользователя и т.д.)
projects/   — файлы проектов (xlsx/csv с карточками проектов)
```

### `alina-rag console` — Локальный чат-бот

Запускает интерактивный чат в терминале. Команды внутри чата:

- `/exit` — выход
- `/clear` — очистить историю
- `/verbose` — показать информацию о поиске

### `alina-rag vk` — VK бот

Запускает бота для сообщества ВКонтакте. Требует `VK_TOKEN` и `VK_GROUP_ID` в `.env`.

### `alina-rag batch` — Массовая обработка вопросов

Обрабатывает CSV/XLSX файлы из папки `tests/`. Файл должен содержать колонки:

| № | Вопрос | Ответ |
|---|---|---|
| 1 | Что такое транзакт? | *(пусто)* |

Система заполняет колонку «Ответ» и сохраняет результат в `*_filled.*`.

### `alina-rag test` — Тестирование с самооценкой

Обрабатывает CSV/XLSX файлы из папки `tests/`. Файл должен содержать колонки:

| № | Вопрос | Ответ | Правильный ответ |
|---|---|---|---|
| 1 | Что такое транзакт? | *(пусто)* | Транзакт — динамический объект GPSS... |

Система:
1. Заполняет ответ (если пустой)
2. Сравнивает ответ с правильным через LLM-судью
3. Выставляет оценку от 1 до 10
4. Сохраняет результат в `*_scored.*`
5. Выводит итоговую статистику в консоль

## Архитектура

```
alina_rag/
    main.py         — CLI (typer): console, vk, index, batch, test
    config.py       — Pydantic Settings
    prompts.py      — Системные промпты
    agent.py        — RAG-цепочка: Qdrant + BM25 → LLM
    indexer.py      — Индексация документов
    console_bot.py  — Терминальный чат
    vk_bot.py       — VK бот
    batch_mode.py   — Массовая обработка
    test_mode.py    — Тестирование с самооценкой
```

## Docker-сервисы

| Сервис | Порт | Описание |
|---|---|---|
| `app` | — | Приложение |
| `ollama` | 11434 | Локальный LLM (qwen2.5:1.5b + nomic-embed-text) |
| `qdrant` | 6333 | Векторная БД |
| `postgres` | 5432 | Реляционная БД (PostgreSQL 18) |

## Инструкция администратора

### Установка

```bash
git clone <repo-url>
cd summer-internship
cp .env.example .env
```

### Настройка

1. Отредактируйте `.env` — укажите токены VK, если нужен бот
2. Поместите документацию в `docs/`
3. Поместите файлы проектов в `projects/`

### Запуск

```bash
docker compose up -d          # запуск всех сервисов
docker compose exec app uv run alina-rag index   # первичная индексация
```

### Обновление данных

При добавлении новых документов:

```bash
docker compose exec app uv run alina-rag index
```

## Инструкция пользователя

### Консольный чат

```bash
uv run alina-rag console
```

Задавайте вопросы по работе с ALINA GPSS в свободной форме. Система ищет релевантные фрагменты в документации и формирует ответ.

### Типы поддерживаемых вопросов

- **Справочные:** «Что такое транзакт?», «Какие блоки есть в GPSS?»
- **Практические:** «Как создать очередь?», «Как настроить блок GENERATE?»
- **По примерам:** «Есть ли пример модели склада?»

### VK бот

Напишите сообщение в сообщество ВК. Бот ответит в течение нескольких секунд.

## Технологический стек

- Python 3.14
- LangChain (RAG-фреймворк)
- Ollama (локальный LLM)
- Qdrant (векторный поиск)
- BM25 (ключевые слова)
- PostgreSQL 18 (история чатов)
- Docker Compose (развёртывание)
