"""Telegram bot for the ALINA GPSS Consultant."""

import logging
from typing import TYPE_CHECKING

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, filters

from alina_rag.domain.models import BotPlatform, UserId

if TYPE_CHECKING:
    from alina_rag.application.chat_service import ChatService

logger = logging.getLogger(__name__)

WELCOME_MESSAGE = (
    "👋 <b>Добро пожаловать в ALINA GPSS AI Консультант!</b>\n\n"
    "Я помогу вам разобраться с GPSS — языком имитационного моделирования. "
    "Задайте вопрос о блоках, синтаксисе, примерах или лучших практиках.\n\n"
    "Просто напишите сообщение — я отвечу!"
)

MAX_MESSAGE_LENGTH = 4000


async def _start_handler(update: Update, _context: object) -> None:
    """Handle /start command."""
    assert update.message is not None
    await update.message.reply_text(
        WELCOME_MESSAGE, parse_mode=ParseMode.HTML
    )


def _build_message_handler(
    chat_service: "ChatService",
) -> "MessageHandler":  # type: ignore[type-arg]
    """Build the text message handler with bound chat service."""

    async def handle_text(update: Update, _context: object) -> None:
        """Handle incoming text messages."""
        assert update.message is not None
        assert update.effective_user is not None

        user_id = UserId(
            BotPlatform.TELEGRAM, str(update.effective_user.id)
        )
        text = update.message.text or ""

        try:
            response = await chat_service.handle_message(user_id, text)
        except Exception:
            logger.exception("Telegram chat service error")
            await update.message.reply_text(
                "⚠️ Произошла ошибка. Попробуйте позже."
            )
            return

        # Split long responses
        if len(response) <= MAX_MESSAGE_LENGTH:
            await update.message.reply_text(
                response, parse_mode=ParseMode.HTML
            )
        else:
            parts = _split_text(response, MAX_MESSAGE_LENGTH)
            for i, part in enumerate(parts):
                prefix = (
                    f"<i>(часть {i + 1}/{len(parts)})</i>\n\n"
                    if i > 0
                    else ""
                )
                await update.message.reply_text(
                    prefix + part, parse_mode=ParseMode.HTML
                )

    return handle_text  # type: ignore[return-value]


def _split_text(text: str, max_len: int) -> list[str]:
    """Split text into chunks not exceeding max_len characters.

    Attempts to split on paragraph boundaries, then sentence boundaries,
    then word boundaries.
    """
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break

        # Try paragraph split
        cut = remaining.rfind("\n\n", 0, max_len)
        if cut < 0:
            # Try sentence split
            cut = _find_sentence_boundary(remaining, max_len)
        if cut < 0:
            # Try word split
            cut = remaining.rfind(" ", 0, max_len)
        if cut < 0:
            cut = max_len

        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    return parts


def _find_sentence_boundary(text: str, max_len: int) -> int:
    """Find the last sentence boundary before max_len."""
    best = -1
    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        pos = text.rfind(sep, 0, max_len)
        if pos > best:
            best = pos
    return best


async def run_telegram(
    chat_service: "ChatService", token: str
) -> None:
    """Start the Telegram bot using python-telegram-bot v21+.

    Blocks until the application is stopped.
    """
    application: Application = ApplicationBuilder().token(token).build()  # type: ignore[type-arg]

    application.add_handler(CommandHandler("start", _start_handler))
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            _build_message_handler(chat_service),  # type: ignore[arg-type]
        )
    )

    logger.info("Telegram bot starting...")
    application.run_polling()
