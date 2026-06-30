"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_test_plan_csv(tmp_path):
    """Create a minimal test plan CSV for testing."""
    csv_path = tmp_path / "test_plan.csv"
    csv_path.write_text(
        "№,Тип вопроса,Вопрос,Критерий корректного ответа,Ответ системы,Результат (корректно/некорректно),Комментарий\n"
        "1,справочный,Что такое транзакт?,Ответ содержит определение транзакта как динамического объекта,,\n"
        "2,практический,Как создать транзакт?,Ответ упоминает блок GENERATE,,\n"
        "3,справочный,Что такое очередь?,Ответ объясняет сбор статистики ожидания,,\n",
        encoding="utf-8",
    )
    return csv_path
