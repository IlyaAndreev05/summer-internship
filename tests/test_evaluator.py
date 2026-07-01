from alina_rag.prompts import JUDGE_PROMPT


class TestJudgePrompt:
    def test_prompt_contains_placeholders(self):
        assert "{question}" in JUDGE_PROMPT
        assert "{criterion}" in JUDGE_PROMPT
        assert "{answer}" in JUDGE_PROMPT

    def test_prompt_format(self):
        result = JUDGE_PROMPT.format(question="Q", criterion="C", answer="A")
        assert "Q" in result
        assert "C" in result
        assert "A" in result
        assert "Вердикт:" in result


class TestJudgeParse:
    def test_parse_correct_russian(self):
        response = "Вердикт: корректно\nКомментарий: ответ содержит определение"
        assert "корректно" in response
        assert "Комментарий" in response

    def test_parse_incorrect_russian(self):
        response = "Вердикт: некорректно\nКомментарий: ответ не соответствует"
        assert "некорректно" in response
