import pytest


@pytest.fixture
def sample_test_csv(tmp_path):
    csv_path = tmp_path / "test_plan.csv"
    csv_path.write_text(
        "Вопрос,Ответ системы\nЧто такое транзакт?,\nКак создать транзакт?,\nЧто такое очередь?,\n",
        encoding="utf-8",
    )
    return csv_path
