import pandas as pd
from rich.console import Console
from rich.table import Table

from alina_rag.agent import RAGAgent
from alina_rag.config import settings

COL_QUESTION = "Вопрос"
COL_ANSWER = "Ответ системы"
COL_CORRECT = "Правильный ответ"
COL_SCORE = "Оценка (1-10)"

SCORE_PROMPT = """Ты — эксперт по оценке качества ответов ИИ-консультанта по GPSS.

Оцени ответ системы на вопрос. Сравни его с правильным ответом и поставь оценку от 1 до 10:
- 10: ответ полностью совпадает с правильным по смыслу, все ключевые моменты освещены.
- 7-9: ответ в целом верный, но упущены некоторые детали.
- 4-6: ответ частично верный, есть неточности или не хватает важной информации.
- 1-3: ответ неверный, не по теме или содержит грубые ошибки.

Выдай ответ строго в формате:
Оценка: N
Комментарий: <одно предложение>

Вопрос: {question}
Правильный ответ: {correct_answer}
Ответ системы: {system_answer}"""


def _read_file(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_excel(path)


def _write_file(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_excel(path, index=False)


def _score_answer(agent, question, correct, system_answer):
    prompt = SCORE_PROMPT.format(
        question=question,
        correct_answer=correct,
        system_answer=system_answer,
    )
    try:
        response = agent.get_llm().invoke(
            [
                {"role": "system", "content": "Ты — эксперт по оценке ответов."},
                {"role": "user", "content": prompt},
            ]
        )
        text = response.content
    except Exception:
        return 0, "Ошибка оценки"

    score = 0
    comment = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("оценка:"):
            try:
                score = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                score = 0
        elif stripped.lower().startswith("комментарий:"):
            comment = stripped.split(":", 1)[1].strip()
    if not comment:
        comment = text[:200]
    return max(0, min(10, score)), comment


def run_tests(agent: RAGAgent):
    input_dir = settings.questions_input_path
    output_dir = settings.questions_output_path

    if not input_dir.exists():
        return

    console = Console()
    all_scores = []

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if "_scored" in path.stem or "_filled" in path.stem:
            continue

        suffix = path.suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls"):
            continue

        df = _read_file(path)
        if COL_QUESTION not in df.columns:
            continue

        if COL_ANSWER not in df.columns:
            df[COL_ANSWER] = ""
        if COL_SCORE not in df.columns:
            df[COL_SCORE] = ""

        has_correct = COL_CORRECT in df.columns

        for idx, row in df.iterrows():
            question = str(row.get(COL_QUESTION, "")).strip()
            if not question or question == "nan":
                continue

            existing_answer = str(row.get(COL_ANSWER, "")).strip()
            if existing_answer and existing_answer != "nan" and existing_answer != "":
                system_answer = existing_answer
            else:
                try:
                    system_answer = agent.answer(question)
                    df.at[idx, COL_ANSWER] = system_answer
                except Exception:
                    df.at[idx, COL_ANSWER] = "ОШИБКА"
                    continue

            if not has_correct:
                continue

            correct = str(row.get(COL_CORRECT, "")).strip()
            if not correct or correct == "nan":
                continue

            score, _comment = _score_answer(agent, question, correct, system_answer)
            df.at[idx, COL_SCORE] = score
            all_scores.append(score)

        out_path = output_dir / f"{path.stem}_scored{path.suffix}"
        _write_file(df, out_path)
        console.print(f"[green]Saved {out_path}[/]")

    if not all_scores:
        console.print("[yellow]No questions with correct answers found for scoring.[/]")
        return

    avg = sum(all_scores) / len(all_scores)
    table = Table(title="Результаты тестирования")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="green")

    table.add_row("Всего вопросов", str(len(all_scores)))
    table.add_row("Средняя оценка", f"{avg:.1f} / 10")
    table.add_row("Оценок 8-10 (отлично)", str(sum(1 for s in all_scores if s >= 8)))
    table.add_row("Оценок 5-7 (средне)", str(sum(1 for s in all_scores if 5 <= s < 8)))
    table.add_row("Оценок 1-4 (плохо)", str(sum(1 for s in all_scores if s < 5)))
    table.add_row(
        "Процент успешных (≥5)",
        f"{sum(1 for s in all_scores if s >= 5) / len(all_scores) * 100:.1f}%",
    )

    console.print()
    console.print(table)
    console.print()
