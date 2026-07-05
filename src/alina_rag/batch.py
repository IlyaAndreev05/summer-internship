import logging
from pathlib import Path

import pandas as pd
from rich.console import Console

from alina_rag.agent import RAGAgent
from alina_rag.config import settings

logger = logging.getLogger(__name__)

COL_QUESTION = "Вопрос"
COL_ANSWER = "Ответ системы"
COL_ANSWER_ALT = "Ответ"


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_excel(path)


def _write_file(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_excel(path, index=False)


def run_batch(agent: RAGAgent) -> None:
    input_dir = settings.questions_input_path
    output_dir = settings.questions_output_path

    if not input_dir.exists():
        logger.error("Questions input directory not found: %s", input_dir)
        return

    console = Console()
    total_processed = 0

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if "_filled" in path.stem or "_scored" in path.stem:
            continue

        suffix = path.suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls"):
            continue

        df = _read_file(path)

        question_col = None
        if COL_QUESTION in df.columns:
            question_col = COL_QUESTION
        elif COL_ANSWER_ALT in df.columns:
            question_col = COL_ANSWER_ALT

        if question_col is None:
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

        out_path = output_dir / f"{path.stem}_filled{path.suffix}"
        _write_file(df, out_path)
        console.print(f"[green]Saved {out_path}[/]")

    console.print(f"\n[bold green]Processed {total_processed} questions[/]")
