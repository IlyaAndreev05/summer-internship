"""VK bot for the ALINA GPSS Consultant."""

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any

from vk_api import VkApi

from alina_rag.domain.models import BotPlatform, UserId

if TYPE_CHECKING:
    from alina_rag.application.chat_service import ChatService

logger = logging.getLogger(__name__)


async def run_vk(
    chat_service: "ChatService", token: str, group_id: str
) -> None:
    """Start the VK bot using Long Poll API.

    Listens for group messages and responds via messages.send.
    Blocks until interrupted.
    """
    vk_session = VkApi(token=token)
    vk = vk_session.get_api()

    longpoll = _VkLongPoll(vk, group_id)
    logger.info("VK bot starting (group_id=%s)...", group_id)

    while True:
        try:
            events = await asyncio.to_thread(longpoll.poll)
        except Exception:
            logger.exception("VK Long Poll error, retrying in 5s...")
            await asyncio.sleep(5)
            continue

        for event in events:
            msg_type: str = str(event.get("type", ""))
            if msg_type != "message_new":
                continue

            obj: dict[str, Any] = event.get("object", {})
            msg: dict[str, Any] = obj.get("message", {})
            # Skip own outbound messages
            if msg.get("out", 0) != 0:
                continue

            user_id_raw = str(msg["from_id"])
            text = msg.get("text", "").strip()
            if not text:
                continue

            user_id = UserId(BotPlatform.VK, user_id_raw)

            try:
                response = await chat_service.handle_message(
                    user_id, text
                )
            except Exception:
                logger.exception("VK chat service error")
                response = "⚠️ Произошла ошибка. Попробуйте позже."

            try:
                await asyncio.to_thread(
                    vk.messages.send,
                    user_id=int(msg["from_id"]),
                    message=response,
                    random_id=random.randint(0, 2**31 - 1),
                )
            except Exception:
                logger.exception("VK send error")


class _VkLongPoll:
    """Thin async-compatible wrapper around VK Bots Long Poll."""

    def __init__(self, vk: Any, group_id: str) -> None:
        self._vk: Any = vk
        self._group_id = int(group_id)
        self._server: str = ""
        self._key: str = ""
        self._ts: str = ""

    def poll(self) -> list[dict[str, Any]]:
        """Poll for new events (blocking, called via asyncio.to_thread)."""
        import json as _json
        import urllib.request

        self._ensure_connected()

        server_url = (
            f"{self._server}?act=a_check&key={self._key}"
            f"&ts={self._ts}&wait=25"
        )
        req = urllib.request.Request(server_url)
        with urllib.request.urlopen(req, timeout=30) as resp_obj:
            data: dict[str, Any] = _json.loads(
                resp_obj.read().decode("utf-8")
            )

        if "failed" in data:
            self._server = ""
            self._key = ""
            self._ts = ""
            return []

        self._ts = str(data.get("ts", self._ts))
        updates: list[dict[str, Any]] = data.get("updates", [])
        return updates

    def _ensure_connected(self) -> None:
        """Connect to the Long Poll server if not already connected."""
        if not self._server:
            lp_data: dict[str, Any] = self._vk.groups.getLongPollServer(
                group_id=self._group_id
            )
            self._server = str(lp_data["server"])
            self._key = str(lp_data["key"])
            self._ts = str(lp_data["ts"])
