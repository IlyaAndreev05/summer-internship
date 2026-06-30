"""LLM-as-Judge evaluator for RAG answer quality."""

import logging

from alina_rag.domain.interfaces import LLMProvider

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Ты — эксперт по оценке качества ответов ИИ-консультанта по системе GPSS.

Оцени ответ системы по следующему критерию. Ответ должен быть оценён как "корректно" или "некорректно".

Правила оценки:
- "корректно" — ответ содержит информацию, соответствующую критерию, даже если не дословно.
- "некорректно" — ответ противоречит критерию, не содержит нужной информации, или содержит явные ошибки.
- Если ответ говорит "я не знаю" или "информация не найдена" — это НЕкорректно (кроме случаев когда вопрос действительно вне темы GPSS).

Выдай ответ строго в формате:

Вердикт: корректно|некорректно
Комментарий: <одно предложение почему>

Вопрос: {question}
Критерий корректного ответа: {criterion}
Ответ системы: {answer}"""


class LLMJudge:
    """Uses an LLM to judge whether a RAG answer matches expected criteria."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def evaluate(
        self, question: str, criterion: str, answer: str
    ) -> tuple[bool, str]:
        """Evaluate one answer.

        Returns:
            (is_correct, comment) tuple.
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            criterion=criterion,
            answer=answer,
        )
        try:
            response = await self._llm.generate(
                system_prompt="Ты — эксперт по оценке качества ответов. Отвечай строго по формату.",
                user_prompt=prompt,
            )
        except Exception as exc:
            logger.warning("Judge LLM call failed: %s", exc)
            return False, f"Ошибка оценки: {exc}"

        verdict, comment = self._parse(response)
        return verdict, comment

    @staticmethod
    def _parse(response: str) -> tuple[bool, str]:
        """Parse the judge's response into (is_correct, comment)."""
        is_correct = False
        comment = ""

        for line in response.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("вердикт:"):
                value = stripped.split(":", 1)[1].strip().lower()
                is_correct = "корректно" in value
            elif stripped.lower().startswith("комментарий:"):
                comment = stripped.split(":", 1)[1].strip()

        if not comment:
            comment = response[:200]

        return is_correct, comment
