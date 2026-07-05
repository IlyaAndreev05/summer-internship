import argparse
import logging

from alina_rag.agent import RAGAgent
from alina_rag.batch import run_batch
from alina_rag.config import settings
from alina_rag.console import run_console
from alina_rag.test_mode import run_tests
from alina_rag.vk_bot import run_vk_bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for _noise in ("httpx", "httpcore", "ollama", "qdrant_client"):
    logging.getLogger(_noise).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ALINA GPSS AI Consultant")
    parser.add_argument("mode", nargs="?", default=None, choices=["console", "vk", "batch", "test"])
    args = parser.parse_args()

    mode = args.mode or settings.app_mode
    logger.info("Starting in %s mode", mode)

    if mode == "vk":
        if not settings.vk_token or not settings.vk_group_id:
            raise SystemExit("VK_TOKEN and VK_GROUP_ID required")
        agent = RAGAgent()
        run_vk_bot(agent, token=settings.vk_token, group_id=settings.vk_group_id)
    elif mode == "batch":
        agent = RAGAgent()
        run_batch(agent)
    elif mode == "console":
        agent = RAGAgent()
        run_console(agent, verbose=settings.chat_verbose)
    elif mode == "test":
        agent = RAGAgent()
        run_tests(agent)
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
