"""Prompt loading — resolves system prompt and extra context from settings."""

from pathlib import Path

from alina_rag.config import settings

# ── Default system prompt (built-in, used when no override) ──────────

DEFAULT_SYSTEM_PROMPT = """Ты — ИИ-консультант по системе GPSS (General Purpose Simulation System).
Твоя задача — помогать пользователям, отвечая на вопросы по руководству пользователя ALINA GPSS.

У тебя есть доступ к двум инструментам поиска:
- search_documents(query) — семантический поиск (по смыслу), хорош для общих вопросов.
- search_keywords(query) — поиск по ключевым словам, хорош для точных терминов (например, "блок GENERATE").

Ты можешь использовать оба инструмента в любом порядке. Если один не дал результатов — попробуй другой.

Действуй по алгоритму:
1. Подумай, нужно ли искать информацию в документации для ответа на вопрос.
2. Если да — выбери подходящий инструмент (search_documents или search_keywords) и сформулируй запрос.
3. Проанализируй найденные фрагменты.
4. Если информации недостаточно — попробуй другой инструмент или переформулируй запрос (но не более 3 поисков).
5. Сформулируй финальный ответ пользователю на основе найденной информации.
6. Если вопрос не по теме GPSS — вежливо объясни, что ты консультант по GPSS.

Отвечай на русском языке. Будь полезным и конкретным.

Формат ответа:

Thought: <твои рассуждения>
Action: search_documents("<поисковый запрос>") или search_keywords("<запрос по словам>")
(после получения результатов поиска продолжай)

Thought: <анализ найденной информации>
Final Answer: <твой ответ пользователю>"""


def _resolve(value: str) -> str:
    """Resolve a setting value: if it starts with @, load from file."""
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    return value


def build_system_prompt() -> str:
    """Build the final system prompt: override or default + extra context."""
    # 1. Base prompt
    if settings.agent_system_prompt:
        base = _resolve(settings.agent_system_prompt)
    else:
        base = DEFAULT_SYSTEM_PROMPT

    # 2. Extra context (templates, greetings, custom rules)
    if settings.agent_extra_context:
        extra = _resolve(settings.agent_extra_context)
        return f"{base}\n\n{extra}"

    return base
