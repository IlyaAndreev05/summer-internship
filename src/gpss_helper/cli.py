import argparse
import logging
import sys

from .bm25_search import BM25Search
from .config import Settings
from .indexer import Indexer
from .modes.batch import BatchMode
from .modes.console import ConsoleMode
from .modes.test import TestMode
from .modes.vk_bot import VkBotMode
from .rag_agent import RAGAgent
from .vector_store import QdrantVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="gpss-helper")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["vk", "batch", "console", "test"],
        default=None,
    )
    parser.add_argument(
        "--mode",
        dest="mode_flag",
        choices=["vk", "batch", "console", "test"],
        default=None,
    )
    args = parser.parse_args()
    mode = args.mode or args.mode_flag or "console"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gpss_helper")

    settings = Settings()
    logger.info("Ollama: %s | Qdrant: %s", settings.ollama_url, settings.qdrant_url)
    vector_store = QdrantVectorStore(settings)
    bm25 = BM25Search()
    indexer = Indexer(settings, vector_store, bm25)
    agent = RAGAgent(settings, vector_store, bm25)

    logger.info("Starting indexing in background...")
    indexer.start()

    if mode == "console":
        ConsoleMode(agent, indexer).run()
    elif mode == "vk":
        VkBotMode(agent, indexer, settings).run()
    elif mode == "batch":
        if not indexer.indexed:
            print("Идёт процесс индексации...")
            sys.exit(0)
        BatchMode(agent, settings).run()
    elif mode == "test":
        if not indexer.indexed:
            print("Идёт процесс индексации...")
            sys.exit(0)
        TestMode(agent, settings).run()
