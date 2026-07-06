from __future__ import annotations

import logging
import random
import time

from vk_api import VkApi

from alina_rag.agent import RAGAgent
from alina_rag.indexer import Indexer
from alina_rag.models import ChatHistory

logger = logging.getLogger(__name__)


def run_vk_bot(agent: RAGAgent, indexer: Indexer, token: str, group_id: str) -> None:
    vk_session = VkApi(token=token)
    vk = vk_session.get_api()

    longpoll = vk_session.method("groups.getLongPollServer", {"group_id": group_id})
    ts = longpoll["ts"]

    histories: dict[int, ChatHistory] = {}

    logger.info("VK bot started (group_id=%s)", group_id)

    while True:
        try:
            response = vk_session.method(
                "messages.getLongPollHistory",
                {
                    "ts": ts,
                    "pts": 0,
                    "fields": "",
                    "events_limit": 1000,
                    "msgs_limit": 100,
                },
            )
        except Exception:
            logger.exception("LongPoll error, reconnecting...")
            time.sleep(5)
            longpoll = vk_session.method("groups.getLongPollServer", {"group_id": group_id})
            ts = longpoll["ts"]
            continue

        ts = response.get("new_ts", ts)

        for event in response.get("updates", []):
            if event.get("type") != "message_new":
                continue

            msg = event.get("object", {}).get("message", {})
            peer_id = msg.get("peer_id", 0)
            text = msg.get("text", "").strip()

            if not text or peer_id == 0:
                continue

            if not indexer.is_ready:
                try:
                    vk.messages.send(
                        peer_id=peer_id,
                        message="⏳ Индексация в процессе, подождите...",
                        random_id=random.randint(0, 2**31 - 1),
                    )
                except Exception:
                    logger.exception("Failed to send indexer-not-ready message")
                continue

            history = histories.setdefault(peer_id, ChatHistory())

            try:
                reply = agent.answer(text, history=history.last_dicts(10))
                history.add_user(text)
                history.add_assistant(reply)

                vk.messages.send(
                    peer_id=peer_id,
                    message=reply[:4096],
                    random_id=random.randint(0, 2**31 - 1),
                )
            except Exception:
                logger.exception("Failed to process message")

        time.sleep(0.5)
