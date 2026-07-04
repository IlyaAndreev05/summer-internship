# ALINA GPSS AI Консультант

ИИ-консультант по системе GPSS (General Purpose Simulation System) на основе RAG (Retrieval-Augmented Generation). Отвечает на вопросы по документации ALINA GPSS, используя семантический и ключевой поиск.

## Архитектура

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Вопрос      │────▶│  RAG Agent  │────▶│  Ответ      │
│  (пользователь)│    │  (ReAct)    │     │  (LLM)      │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Qdrant  │ │ PostgreSQL│ │  Ollama  │
        │ (векторы) │ │ (чанки + │ │  (LLM +  │
        │          │ │ trigram + │ │ embed)   │
        │          │ │ BM25)    │ │          │
        └──────────┘ └──────────┘ └──────────┘
```

### Как работает поиск

1. Пользователь задаёт вопрос.
2. LLM решает, какой инструмент поиска использовать (семантический или по ключевым словам).
3. Система ищет релевантные фрагменты в Qdrant (векторный поиск) и PostgreSQL (trigram + BM25).
4. LLM генерирует ответ на основе найденных фрагментов.
5. Если информации недостаточно, LLM может запросить повторный поиск другими методами (до 5 итераций).

## Быстрый старт

### 1. Клонировать и настроить

```bash
git clone <url-репозитория>
cd summer-internship
cp .env.example .env
```

### 2. Запустить контейнеры

```bash
docker compose up -d
```

### 3. Дождаться загрузки моделей

```bash
docker compose logs -f ollama
```

Ollama автоматически скачает модели `qwen2.5:1.5b` и `nomic-embed-text` (1–3 минуты). Нажмите `Ctrl+C` после загрузки.

### 4. Задать вопрос

```bash
docker compose exec -it app uv run alina-rag console
```

```
Вы: Что такое транзакт в GPSS?
```

Индексация документов происходит **автоматически** при каждом запуске контейнера `app`. Система проверяет хеши файлов и переиндексирует только новые или изменённые.

## Режимы работы

| Режим | Запуск | Описание |
|---|---|---|
| `console` | `docker compose exec -it app uv run alina-rag console` | Интерактивный чат в терминале |
| `batch` | `docker compose exec app uv run alina-rag batch` | Пакетная обработка вопросов из файлов в `tests/` |
| `vk` | `docker compose exec app uv run alina-rag vk` | VK-бот (требуются `VK_TOKEN` и `VK_GROUP_ID`) |
| `test` | `docker compose exec app uv run alina-rag test` | Тестирование с самооценкой (LLM-судья, 1-10) |

Команды внутри консоли: `/exit`, `/clear`, `/verbose`.

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `APP_MODE` | `console` | Режим: `console`, `vk`, `batch` |
| `LLM_PROVIDER` | `ollama` | Провайдер LLM |
| `LLM_MODEL` | `qwen2.5:1.5b` | Модель генерации ответов |
| `LLM_BASE_URL` | `http://ollama:11434` | URL Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Модель эмбеддингов |
| `QDRANT_URL` | `http://qdrant:6333` | URL Qdrant |
| `QDRANT_COLLECTION` | `alina_docs` | Коллекция Qdrant |
| `POSTGRES_URL` | `postgresql://alina:alina@postgres:5432/alina` | PostgreSQL |
| `VK_TOKEN` | — | Токен VK (для `vk` режима) |
| `VK_GROUP_ID` | — | ID группы VK (для `vk` режима) |
| `DOCS_DIR` | `docs` | Папка с документацией |
| `PROJECTS_DIR` | `projects` | Папка с проектами |
| `TESTS_DIR` | `tests` | Папка с тестами |
| `CHUNK_SIZE` | `500` | Размер чанка (символы) |
| `CHUNK_OVERLAP` | `100` | Перекрытие чанков (символы) |
| `CHAT_MAX_MESSAGES` | `20` | Лимит сообщений в истории |
| `CHAT_VERBOSE` | `false` | Отладка поиска |

## Docker-сервисы

| Сервис | Образ | Порт | Память | Описание |
|---|---|---|---|---|
| `app` | Сборка из `Dockerfile` | — | 2 ГБ | Python-приложение |
| `ollama` | `ollama/ollama:0.31.1` | 11434 | 8 ГБ | LLM и эмбеддинги |
| `qdrant` | `qdrant/qdrant:v1.18.2` | 6333 | 1 ГБ | Векторная БД |
| `postgres` | `postgres:18` | 5432 | 512 МБ | Хранение чанков и trigram-поиск |

Все сервисы перезапускаются автоматически (`restart: unless-stopped`).

## Структура проекта

```
├── src/alina_rag/
│   ├── main.py           — точка входа (CLI)
│   ├── config.py          — настройки (pydantic-settings)
│   ├── agent.py           — RAG-агент: векторный + BM25 + trigram поиск → LLM
│   ├── indexer.py         — загрузчики документов (PDF, DOCX, XLSX, CSV, TXT, MD)
│   ├── auto_index.py      — автоматическая индексация с дедупликацией
│   ├── db.py              — PostgreSQL: чанки, trigram-поиск, хеши файлов
│   ├── prompts.py         — системные промпты для LLM и LLM-судьи
│   ├── console_bot.py     — интерактивная консоль (Rich)
│   ├── vk_bot.py          — VK LongPoll бот
│   └── batch_mode.py      — пакетная обработка вопросов
├── docs/
│   ├── admin_guide.md     — инструкция администратора
│   └── user_guide.md      — инструкция пользователя
├── projects/              — файлы проектов (.xlsx, .csv)
├── tests/
│   └── test_plan.csv      — тестовый план (50 вопросов)
├── docker-compose.yml     — оркестрация сервисов
├── Dockerfile             — сборка приложения
├── docker-entrypoint.sh   — авто-индексация + запуск
├── .env.example           — шаблон конфигурации
└── pyproject.toml         — зависимости Python
```

## Стек технологий

- **Python 3.14** — основной язык
- **LangChain** — фреймворк RAG-агента
- **Ollama** — LLM-сервер (модель `qwen2.5:1.5b` + эмбеддинги `nomic-embed-text`)
- **Qdrant** — векторная БД для семантического поиска
- **PostgreSQL 18** — хранение чанков, trigram-поиск, BM25
- **Docker Compose** — контейнеризация и оркестрация
- **Rich** — форматированный вывод в консоли
- **pandas** — обработка табличных данных

## Локальная разработка (без Docker)

Требования: Python 3.12+, [uv](https://docs.astral.sh/uv/), запущенные Ollama, Qdrant и PostgreSQL.

```bash
cp .env.example .env
# Отредактируйте .env: замените адреса сервисов на localhost
# LLM_BASE_URL=http://localhost:11434
# QDRANT_URL=http://localhost:6333
# POSTGRES_URL=postgresql://alina:alina@localhost:5432/alina

uv sync
uv run alina-rag console
```

## Документация

- [Инструкция администратора](docs/admin_guide.md) — установка, настройка, управление
- [Инструкция пользователя](docs/user_guide.md) — как задавать вопросы и интерпретировать ответы

## Тестирование

Тестовый план с 50 вопросами по GPSS находится в `tests/test_plan.csv`.

### Автоматическое тестирование (LLM-судья)

```bash
docker compose exec app uv run alina-rag test
```

Режим `test` читает вопросы из `tests/`, генерирует ответы агента, затем LLM-судья сравнивает с эталоном и ставит оценку 1–10. Результат: `*_scored.*` в `tests/` + статистика в консоли.

Формат тестового плана: CSV с колонками `№`, `Тип вопроса`, `Вопрос`, `Критерий корректного ответа`, `Ответ системы`, `Результат`, `Комментарий`.
