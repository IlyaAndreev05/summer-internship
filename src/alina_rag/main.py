from __future__ import annotations

import argparse
import logging

from alina_rag.agent import RAGAgent
from alina_rag.batch import run_batch
from alina_rag.config import settings
from alina_rag.console import run_console
from alina_rag.db import Database
from alina_rag.indexer import Indexer
from alina_rag.test_mode import run_tests
from alina_rag.vk_bot import run_vk_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for _noise in ("httpx", "httpcore", "ollama", "qdrant_client"):
    logging.getLogger(_noise).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ALINA GPSS AI Consultant")
    parser.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=["console", "vk", "batch", "test"],
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    mode = args.mode or settings.app_mode
    verbose = args.verbose or settings.chat_verbose
    logger.info("Starting in %s mode", mode)

    if mode == "vk" and (not settings.vk_token or not settings.vk_group_id):
        raise SystemExit("VK_TOKEN and VK_GROUP_ID required")

    db = Database(settings.postgres_url)
    indexer = Indexer(
        db=db,
        qdrant_url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        embed_model=settings.embed_model,
        ollama_host=settings.ollama_host,
    )
    agent = RAGAgent(db=db, cfg=settings)

    stats = indexer.index(
        settings.docs_path,
        settings.projects_path,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    if stats:
        logger.info(
            "Indexed: %d files, %d chunks in %.1fs",
            stats.total_files,
            stats.total_chunks,
            stats.elapsed,
        )

    if mode == "console":
        run_console(agent, indexer, verbose=verbose)
    elif mode == "vk":
        run_vk_bot(agent, indexer, token=settings.vk_token, group_id=settings.vk_group_id)
    elif mode == "batch":
        run_batch(agent, indexer, cfg=settings)
    elif mode == "test":
        run_tests(agent, indexer, cfg=settings)
    else:
        raise SystemExit(f"Unknown mode: {mode}")
