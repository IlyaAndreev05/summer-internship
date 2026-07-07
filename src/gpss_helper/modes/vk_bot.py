import logging
import sys

import vk_api
from vk_api.longpoll import VkEventType, VkLongPoll

from ..config import Settings
from ..indexer import Indexer
from ..rag_agent import RAGAgent

logger = logging.getLogger(__name__)


class VkBotMode:
    def __init__(self, agent: RAGAgent, indexer: Indexer, settings: Settings):
        self.agent = agent
        self.indexer = indexer
        self.settings = settings

    def run(self):
        if not self.settings.vk_token:
            logger.error("VK_TOKEN not set")
            sys.exit(1)

        group_id = self.settings.vk_group_id
        if not group_id:
            logger.error("VK_GROUP_ID not set")
            sys.exit(1)

        vk_session = vk_api.VkApi(token=self.settings.vk_token)
        vk = vk_session.get_api()
        longpoll = VkLongPoll(vk_session, group_id=group_id)

        logger.info("VK bot started, listening for messages")

        for event in longpoll.listen():
            if event.type != VkEventType.MESSAGE_NEW or not event.to_me:
                continue
            if event.text is None:
                continue

            if not self.indexer.indexed:
                if self.indexer.error:
                    vk.messages.send(
                        user_id=event.user_id,
                        message="Сервис временно недоступен",
                        random_id=0,
                    )
                    logger.critical("Indexing failed, shutting down bot")
                    sys.exit(1)
                vk.messages.send(
                    user_id=event.user_id,
                    message="Идёт процесс индексации...",
                    random_id=0,
                )
                continue

            answer = self.agent.answer(event.text)
            vk.messages.send(
                user_id=event.user_id,
                message=answer,
                random_id=0,
            )
