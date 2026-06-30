"""Unit tests for the evaluation runner (CSV parsing, output)."""

import csv
from pathlib import Path

import pytest
from alina_rag.evaluation.runner import _read_csv, _write_csv, COL_NUM, COL_QUESTION, COL_RESULT


class TestReadCSV:
    def test_read_valid_csv(self, sample_test_plan_csv):
        rows = _read_csv(sample_test_plan_csv)
        assert len(rows) == 3
        assert rows[0][COL_QUESTION] == "Что такое транзакт?"
        assert rows[1][COL_QUESTION] == "Как создать транзакт?"

    def test_read_missing_file(self):
        rows = _read_csv(Path("/nonexistent/test.csv"))
        assert rows == []

    def test_read_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text(
            "№,Тип вопроса,Вопрос,Критерий корректного ответа,Ответ системы,Результат,Комментарий\n",
            encoding="utf-8",
        )
        rows = _read_csv(csv_path)
        assert rows == []


class TestWriteCSV:
    def test_write_and_reread(self, tmp_path):
        output = tmp_path / "report.csv"
        rows = [
            {
                COL_NUM: "1",
                "Тип вопроса": "справочный",
                COL_QUESTION: "Что такое транзакт?",
                "Критерий корректного ответа": "Определение",
                "Ответ системы": "Транзакт — это...",
                COL_RESULT: "корректно",
                "Комментарий": "ok",
            },
            {
                COL_NUM: "2",
                "Тип вопроса": "практический",
                COL_QUESTION: "Как создать?",
                "Критерий корректного ответа": "GENERATE",
                "Ответ системы": "Не знаю",
                COL_RESULT: "некорректно",
                "Комментарий": "нет ответа",
            },
        ]
        _write_csv(output, rows)

        # Reread and verify
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            reread = list(reader)

        assert len(reread) == 2
        assert reread[0][COL_RESULT] == "корректно"
        assert reread[1][COL_RESULT] == "некорректно"


class TestRunEvaluation:
    """End-to-end test of the evaluation pipeline with a mock agent."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_test_plan_csv, tmp_path):
        from alina_rag.evaluation.runner import run_evaluation

        output = tmp_path / "report.csv"

        class FakeAgent:
            async def process_message(self, user_id, message_text: str) -> str:
                if "транзакт" in message_text.lower():
                    return "Транзакт — динамический объект GPSS."
                if "GENERATE" in message_text.lower() or "создать" in message_text.lower():
                    return "Используйте блок GENERATE."
                return "Информация не найдена."

        class FakeLLM:
            async def generate(self, system_prompt: str, user_prompt: str) -> str:
                # Always return "correct" for testing
                return "Вердикт: корректно\nКомментарий: ответ соответствует критерию"

        agent = FakeAgent()
        llm = FakeLLM()

        summary = await run_evaluation(agent, llm, sample_test_plan_csv, output, delay=0)

        assert summary["total"] == 3
        assert summary["correct"] == 3
        assert summary["percent"] == 100.0
        assert output.exists()

        # Verify output CSV has the right columns
        rows = _read_csv(output)
        assert len(rows) == 3
        for row in rows:
            assert row[COL_RESULT] == "корректно"
            assert row["Ответ системы"] != ""
