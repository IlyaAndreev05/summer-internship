import logging
import random
import time

from vk_api import VkApi

from alina_rag.agent import RAGAgent

logger = logging.getLogger(__name__)


def run_vk_bot(agent: RAGAgent, token: str, group_id: str) -> None:
    """Запуск VK-бота для ответов на сообщения в группе."""
    vk_session = VkApi(token=token)
    vk = vk_session.get_api()

    longpoll = vk_session.method("groups.getLongPollServer", {"group_id": group_id})
    ts = longpoll["ts"]

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

            try:
                reply = agent.answer(text)

                vk.messages.send(
                    peer_id=peer_id,
                    message=reply[:4096],
                    random_id=random.randint(0, 2**31 - 1),
                )
            except Exception:
                logger.exception("Failed to process message")

        time.sleep(0.5)
