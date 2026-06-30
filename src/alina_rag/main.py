"""ALINA GPSS AI Consultant — CLI entry point."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from alina_rag.config import settings
from alina_rag.domain.interfaces import (
    ChatRepository,
    EmbeddingProvider,
    LLMProvider,
    SessionStore,
    VectorStore,
)

logger = logging.getLogger("alina_rag")


class Mode(str, Enum):
    CONSOLE = "console"
    TELEGRAM = "telegram"
    VK = "vk"
    API = "api"
    ALL = "all"


app = typer.Typer(name="alina-rag", help="ALINA GPSS AI Consultant")
main = app  # pyproject.toml entry point


# ── Infrastructure factories ──────────────────────────


def _build_llm_provider() -> LLMProvider:
    """Select and initialise the LLM provider from settings."""
    if settings.llm_provider == "ollama":
        from alina_rag.infrastructure.ollama_provider import OllamaProvider

        return OllamaProvider()
    else:
        from alina_rag.infrastructure.openai_provider import OpenAIProvider

        return OpenAIProvider()


def _build_embedding_provider() -> EmbeddingProvider:
    from alina_rag.infrastructure.embedding_provider import STEmbeddingProvider

    return STEmbeddingProvider()


def _build_vector_store(embed_provider: EmbeddingProvider) -> VectorStore:
    from alina_rag.infrastructure.chroma_store import ChromaStore

    return ChromaStore(settings, embed_provider)


def _build_chat_repo() -> ChatRepository:
    from alina_rag.infrastructure.sqlite_repo import SqliteChatRepository

    return SqliteChatRepository()



def _build_session_store() -> SessionStore:
    from alina_rag.infrastructure.session_store import InMemorySessionStore

    return InMemorySessionStore()


def _build_bm25_store() -> Any:
    from alina_rag.infrastructure.bm25_store import BM25Store

    return BM25Store()


def _build_chunker(embed_provider: Any = None) -> Any:
    """Create the chunker based on settings.chunker_type."""
    from alina_rag.infrastructure.chunker import ParagraphChunker, SemanticChunker

    if settings.chunker_type == "semantic" and embed_provider is not None:
        return SemanticChunker(embed_provider)
    return ParagraphChunker(
        chunk_size=settings.chunker_size,
        overlap=settings.chunker_overlap,
    )


def _build_doc_service(
    vector_store: VectorStore,
    embed_provider: EmbeddingProvider,
    bm25_store: Any = None,
    chunker: Any = None,
) -> Any:
    from alina_rag.application.document_service import DocumentService

    return DocumentService(vector_store, embed_provider, bm25_store, chunker)


def _build_chat_service(
    agent: Any,
    chat_repo: ChatRepository,
) -> Any:
    from alina_rag.application.chat_service import ChatService

    return ChatService(agent, chat_repo)


def _build_agent(
    llm: LLMProvider,
    vector_store: VectorStore,
    session_store: SessionStore,
    bm25_store: Any = None,
) -> Any:
    from alina_rag.application.agent_service import AgentService

    return AgentService(llm, vector_store, bm25_store, session_store)


# ── Commands ──────────────────────────────────────────


@app.command()
def run(
    mode: Mode = typer.Option(Mode.CONSOLE, help="Runtime mode(s)"),
) -> None:
    """Run the ALINA GPSS AI Consultant."""
    asyncio.run(_run(mode))


async def _run(mode: Mode) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Initialising ALINA RAG (mode=%s)", mode.value)

    # 1. Infrastructure
    logger.info("Building LLM provider (%s)", settings.llm_provider)
    llm = _build_llm_provider()

    logger.info("Building embedding provider")
    embed = _build_embedding_provider()

    # Warm up models — download/pull now, not on first question
    logger.info("Warming up LLM (may download model)...")
    await llm._ensure_model()
    logger.info("Warming up embedding model (may download)...")
    await embed._ensure_model()

    logger.info("Building vector store")
    vector = _build_vector_store(embed)

    logger.info("Building chat repository")
    chat_repo = _build_chat_repo()

    logger.info("Building BM25 keyword store")
    bm25 = _build_bm25_store()

    logger.info("Building session store")
    session_store = _build_session_store()
    # 2. Application services
    logger.info("Building agent service")
    agent = _build_agent(llm, vector, session_store, bm25)

    logger.info("Building chat service")
    chat_service = _build_chat_service(agent, chat_repo)

    logger.info("Building document service")
    chunker = _build_chunker(embed)
    doc_service = _build_doc_service(vector, embed, bm25, chunker)
    docs_path = settings.docs_path
    if docs_path.exists():
        logger.info("Auto-ingesting documents from %s", docs_path)
        try:
            if docs_path.is_dir():
                count = await doc_service.ingest_directory(docs_path)
            else:
                count = await doc_service.ingest_file(docs_path)
            logger.info("Ingested %d chunks", count)
        except Exception:
            logger.exception("Document ingestion failed — continuing")
    else:
        logger.info("Documents directory %s not found, skipping auto-ingest", docs_path)

    # 4. Start modes
    tasks: list[asyncio.Task[None]] = []

    if mode in (Mode.CONSOLE, Mode.ALL):
        from alina_rag.presentation.console import run_console

        tasks.append(asyncio.create_task(_safe_task("console", run_console(chat_service))))

    if mode in (Mode.TELEGRAM, Mode.ALL):
        if not settings.telegram_token:
            logger.warning("TELEGRAM_TOKEN not set — skipping Telegram mode")
        else:
            from alina_rag.presentation.telegram_bot import run_telegram

            tasks.append(
                asyncio.create_task(
                    _safe_task(
                        "telegram",
                        run_telegram(chat_service, settings.telegram_token),
                    )
                )
            )

    if mode in (Mode.VK, Mode.ALL):
        if not settings.vk_token or not settings.vk_group_id:
            logger.warning("VK_TOKEN/VK_GROUP_ID not set — skipping VK mode")
        else:
            from alina_rag.presentation.vk_bot import run_vk

            tasks.append(
                asyncio.create_task(
                    _safe_task(
                        "vk",
                        run_vk(chat_service, settings.vk_token, settings.vk_group_id),
                    )
                )
            )

    if mode in (Mode.API, Mode.ALL):
        from alina_rag.presentation.api import run_api

        tasks.append(asyncio.create_task(_safe_task("api", run_api(chat_service, doc_service))))

    if not tasks:
        logger.warning("No modes started — check configuration")
        return

    logger.info("Starting %d mode(s)", len(tasks))
    await asyncio.gather(*tasks)


async def _safe_task(name: str, coro: Any) -> None:
    """Wrap a mode coroutine so one failure doesn't crash the whole app."""
    try:
        await coro
    except asyncio.CancelledError:
        logger.info("Mode %s cancelled", name)
    except Exception:
        logger.exception("Mode %s crashed", name)


@app.command()
def ingest(
    path: str = typer.Option(settings.docs_dir, help="File or directory to ingest"),
) -> None:
    """Ingest documents into the knowledge base."""
    asyncio.run(_ingest(path))


async def _ingest(path: str) -> None:
    logging.basicConfig(level=logging.INFO)

    p = Path(path)
    if not p.exists():
        typer.echo(f"Path not found: {path}", err=True)
        raise typer.Exit(code=1)

    embed = _build_embedding_provider()
    vector = _build_vector_store(embed)
    bm25 = _build_bm25_store()
    chunker = _build_chunker(embed)
    doc_service = _build_doc_service(vector, embed, bm25, chunker)

    if p.is_file():
        count = await doc_service.ingest_file(p)
    else:
        count = await doc_service.ingest_directory(p)

    typer.echo(f"Ingested {count} chunks")


@app.command()
def evaluate(
    input_path: str = typer.Option(
        "tests/test_plan.csv", "--input", "-i",
        help="Path to test plan CSV",
    ),
    output_path: str = typer.Option(
        "tests/report.csv", "--output", "-o",
        help="Where to write the scored report",
    ),
    delay: float = typer.Option(
        1.0, "--delay", "-d",
        help="Seconds between questions",
    ),
) -> None:
    """Evaluate agent answer quality against a test plan CSV."""
    asyncio.run(_evaluate(input_path, output_path, delay))


async def _evaluate(input_path: str, output_path: str, delay: float) -> None:
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path

    from alina_rag.evaluation.runner import run_evaluation

    typer.echo("Initialising agent and judge...")

    embed = _build_embedding_provider()
    vector = _build_vector_store(embed)
    bm25 = _build_bm25_store()
    llm = _build_llm_provider()
    session_store = _build_session_store()
    agent = _build_agent(llm, vector, session_store, bm25)

    csv_in = Path(input_path)
    csv_out = Path(output_path)

    if not csv_in.exists():
        typer.echo(f"Test plan not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Running evaluation: {csv_in} → {csv_out}")
    summary = await run_evaluation(agent, llm, csv_in, csv_out, delay=delay)

    typer.echo(f"\n{'='*50}")
    typer.echo(f"Результаты: {summary['correct']}/{summary['total']} корректных "
                f"({summary['percent']:.1f}%)")
    typer.echo(f"Отчёт сохранён: {csv_out}")

    if summary["percent"] < 70.0:
        typer.echo("⚠️  Качество ниже целевого порога 70% — требуется доработка.")
    else:
        typer.echo("✓  Качество выше целевого порога 70%.")


if __name__ == "__main__":
    app()
