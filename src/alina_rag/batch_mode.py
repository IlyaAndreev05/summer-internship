import logging
from pathlib import Path

import pandas as pd
from rich.console import Console

from alina_rag.agent import RAGAgent
from alina_rag.config import settings

logger = logging.getLogger(__name__)

COL_NUM = "№"
COL_QUESTION = "Вопрос"
COL_ANSWER = "Ответ системы"
COL_ANSWER_ALT = "Ответ"


def _read_file(path: Path) -> pd.DataFrame:
    """Чтение файла вопросов (CSV или Excel)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_excel(path)


def _write_file(df: pd.DataFrame, path: Path) -> None:
    """Запись DataFrame в файл (CSV или Excel)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_excel(path, index=False)


def run_batch(agent: RAGAgent) -> None:
    """Пакетная обработка вопросов из файлов: заполняет ответы агента."""
    tests_dir = settings.tests_path
    if not tests_dir.exists():
        logger.error("Tests directory not found: %s", tests_dir)
        return

    console = Console()
    total_processed = 0

    for path in sorted(tests_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if "_filled" in path.stem or "_scored" in path.stem:
            continue

        suffix = path.suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls"):
            continue

        df = _read_file(path)

        question_col = None
        if COL_QUESTION in df.columns or (COL_NUM in df.columns and COL_ANSWER_ALT in df.columns):
            question_col = COL_QUESTION

        if question_col is None or question_col not in df.columns:
            logger.warning("No question column in %s, skipping", path.name)
            continue

        answer_col = COL_ANSWER if COL_ANSWER in df.columns else COL_ANSWER_ALT
        if answer_col not in df.columns:
            df[answer_col] = ""

        for idx, row in df.iterrows():
            question = str(row.get(question_col, "")).strip()
            if not question or question == "nan":
                continue

            existing = str(row.get(answer_col, "")).strip()
            if existing and existing != "nan" and existing != "":
                continue

            try:
                answer = agent.answer(question)
                df.at[idx, answer_col] = answer
                total_processed += 1
                logger.info("Q%d answered", idx + 1)
            except Exception:
                logger.exception("Failed to answer Q%d", idx + 1)
                df.at[idx, answer_col] = "ОШИБКА"

        out_path = path.parent / f"{path.stem}_filled{path.suffix}"
        _write_file(df, out_path)
        console.print(f"[green]Saved {out_path}[/]")

    console.print(f"\n[bold green]Processed {total_processed} questions[/]")
