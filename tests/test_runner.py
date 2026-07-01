import csv

from alina_rag.test_mode import COL_ANSWER, COL_QUESTION


class TestCSVProcessing:
    def test_col_constants(self):
        assert COL_QUESTION == "Вопрос"
        assert COL_ANSWER == "Ответ системы"


class TestCSVIO:
    def test_read_valid_csv(self, sample_test_csv):
        with open(sample_test_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0][COL_QUESTION] == "Что такое транзакт?"

    def test_write_and_reread(self, tmp_path):
        output = tmp_path / "report.csv"
        fieldnames = [COL_QUESTION, COL_ANSWER]
        rows = [
            {COL_QUESTION: "Q1", COL_ANSWER: "A1"},
            {COL_QUESTION: "Q2", COL_ANSWER: "A2"},
        ]
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            reread = list(reader)

        assert len(reread) == 2
        assert reread[0][COL_ANSWER] == "A1"
