"""Test plan runner — evaluates agent answers against a CSV test plan."""

import csv
import logging
import time
from pathlib import Path

from alina_rag.config import settings
from alina_rag.domain.models import BotPlatform, Message, Role, UserId
from alina_rag.domain.interfaces import LLMProvider
from alina_rag.evaluation.evaluator import LLMJudge

logger = logging.getLogger(__name__)

# CSV columns expected in the test plan
COL_NUM = "№"
COL_TYPE = "Тип вопроса"
COL_QUESTION = "Вопрос"
COL_CRITERION = "Критерий корректного ответа"
COL_ANSWER = "Ответ системы"
COL_RESULT = "Результат (корректно/некорректно)"
COL_COMMENT = "Комментарий"


async def run_evaluation(
    agent,            # AgentService
    llm: LLMProvider,  # for the judge (can be same or different model)
    csv_path: Path,
    output_path: Path,
    delay: float = 1.0,
) -> dict:
    """Run the full evaluation pipeline.

    Args:
        agent: The AgentService instance to test.
        llm: LLM provider for the judge.
        csv_path: Path to the input CSV test plan.
        output_path: Where to write the scored CSV.
        delay: Seconds to wait between questions (avoid rate limiting).

    Returns:
        Dict with summary statistics.
    """
    judge = LLMJudge(llm)
    test_user = UserId(platform=BotPlatform.CONSOLE, platform_user_id="evaluator")

    # Read test plan
    rows = _read_csv(csv_path)
    if not rows:
        logger.error("No questions found in %s", csv_path)
        return {"total": 0, "correct": 0, "percent": 0.0}

    total = 0
    correct = 0
    results: list[dict] = []

    logger.info("Evaluating %d questions...", len(rows))

    for row in rows:
        question = row.get(COL_QUESTION, "").strip()
        criterion = row.get(COL_CRITERION, "").strip()
        if not question:
            continue

        total += 1
        logger.info("[%d/%d] %s", total, len(rows), question[:80])

        # Get agent answer
        try:
            answer = await agent.process_message(test_user, question)
        except Exception as exc:
            logger.error("Agent failed on question: %s", exc)
            answer = f"ERROR: {exc}"

        # Judge the answer
        is_correct, comment = await judge.evaluate(question, criterion, answer)

        if is_correct:
            correct += 1

        result_str = "корректно" if is_correct else "некорректно"
        results.append({
            COL_NUM: row.get(COL_NUM, str(total)),
            COL_TYPE: row.get(COL_TYPE, ""),
            COL_QUESTION: question,
            COL_CRITERION: criterion,
            COL_ANSWER: answer,
            COL_RESULT: result_str,
            COL_COMMENT: comment,
        })

        logger.info("  → %s | %s", result_str, comment[:100])

        if delay > 0:
            time.sleep(delay)

    # Write output
    _write_csv(output_path, results)

    percent = (correct / total * 100) if total > 0 else 0.0
    summary = {"total": total, "correct": correct, "percent": percent}
    logger.info("Evaluation complete: %d/%d correct (%.1f%%)", correct, total, percent)
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read test plan CSV into list of dicts."""
    if not path.exists():
        logger.error("Test plan not found: %s", path)
        return []

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write scored results to CSV."""
    fieldnames = [
        COL_NUM, COL_TYPE, COL_QUESTION, COL_CRITERION,
        COL_ANSWER, COL_RESULT, COL_COMMENT,
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
