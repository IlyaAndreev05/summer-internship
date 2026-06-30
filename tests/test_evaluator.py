"""Unit tests for LLM-as-Judge evaluator."""

import pytest
from alina_rag.evaluation.evaluator import LLMJudge


class FakeLLM:
    """Mock LLM that returns a pre-programmed response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self._response

    async def generate_with_history(self, system_prompt: str, messages: list) -> str:
        return self._response


class TestLLMJudgeParse:
    """Tests for the static _parse method of LLMJudge."""

    def test_parse_correct_russian(self):
        response = "Вердикт: корректно\nКомментарий: ответ содержит определение транзакта"
        is_correct, comment = LLMJudge._parse(response)
        assert is_correct is True
        assert "определение" in comment

    def test_parse_incorrect_russian(self):
        response = "Вердикт: некорректно\nКомментарий: ответ не соответствует критерию"
        is_correct, comment = LLMJudge._parse(response)
        assert is_correct is False
        assert "не соответствует" in comment

    def test_parse_case_insensitive(self):
        response = "ВЕРДИКТ: Корректно\nКомментарий: ok"
        is_correct, _ = LLMJudge._parse(response)
        assert is_correct is True

    def test_parse_extra_text_before_verdict(self):
        response = "Какой-то мусор перед\nВердикт: корректно\nКомментарий: всё хорошо"
        is_correct, _ = LLMJudge._parse(response)
        assert is_correct is True

    def test_parse_missing_comment(self):
        response = "Вердикт: некорректно"
        is_correct, comment = LLMJudge._parse(response)
        assert is_correct is False
        assert comment == "некорректно" or len(comment) > 0  # falls back to raw text

    def test_parse_empty_response(self):
        is_correct, _ = LLMJudge._parse("")
        assert is_correct is False

    def test_parse_english_verdict(self):
        """Ensure only Russian 'корректно' triggers True, not English."""
        response = "Вердикт: correct\nКомментарий: test"
        is_correct, _ = LLMJudge._parse(response)
        assert is_correct is False  # "correct" != "корректно"


class TestLLMJudgeEvaluate:
    """Tests for the async evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_correct(self):
        llm = FakeLLM("Вердикт: корректно\nКомментарий: ответ полный")
        judge = LLMJudge(llm)  # type: ignore[arg-type]
        is_correct, comment = await judge.evaluate(
            "Что такое транзакт?",
            "Определение транзакта",
            "Транзакт — динамический объект GPSS",
        )
        assert is_correct is True
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_evaluate_incorrect(self):
        llm = FakeLLM("Вердикт: некорректно\nКомментарий: ответ пустой")
        judge = LLMJudge(llm)  # type: ignore[arg-type]
        is_correct, _ = await judge.evaluate("Вопрос", "Критерий", "Не знаю")
        assert is_correct is False

    @pytest.mark.asyncio
    async def test_evaluate_llm_failure(self):
        """Judge should return False when LLM call fails."""
        llm = FakeLLM("garbage without verdict")
        judge = LLMJudge(llm)  # type: ignore[arg-type]
        is_correct, _ = await judge.evaluate("Q", "C", "A")
        assert is_correct is False  # no "корректно" in garbage
