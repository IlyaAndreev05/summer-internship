"""SQLite-based chat history repository using aiosqlite."""

from pathlib import Path

import aiosqlite

from alina_rag.config import settings
from alina_rag.domain.interfaces import ChatRepository
from alina_rag.domain.models import Message, Role, UserId


def _extract_db_path(database_url: str) -> Path:
    """Extract filesystem path from a sqlite+aiosqlite:///… connection URL."""
    return Path(database_url.split(":///", 1)[-1])


class SqliteChatRepository(ChatRepository):
    """Chat history persisted in a local SQLite database."""

    def __init__(self) -> None:
        self._db_path = _extract_db_path(settings.database_url)

    async def _ensure_table(self, db: aiosqlite.Connection) -> None:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await db.commit()

    async def save_message(self, message: Message) -> None:
        """Persist a single chat message."""
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            await db.execute(
                "INSERT INTO messages (id, user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    message.id,
                    str(message.user_id) if message.user_id else "",
                    message.role.value,
                    message.content,
                    message.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_history(self, user_id: UserId, limit: int = 20) -> list[Message]:
        """Retrieve the most recent messages for a user."""
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            cursor = await db.execute(
                "SELECT id, user_id, role, content, created_at FROM messages "
                "WHERE user_id = ? ORDER BY created_at ASC LIMIT ?",
                (str(user_id), limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_message(row) for row in rows]

    async def clear_history(self, user_id: UserId) -> None:
        """Remove all messages for a user."""
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            await db.execute("DELETE FROM messages WHERE user_id = ?", (str(user_id),))
            await db.commit()


def _row_to_message(row: tuple[str, str, str, str, str]) -> Message:
    from datetime import datetime

    _id, _user_id, role_str, content, created_at_str = row
    return Message(
        id=_id,
        user_id=_user_id_from_str(_user_id) if _user_id else None,
        role=Role(role_str),
        content=content,
        created_at=datetime.fromisoformat(created_at_str),
    )


def _user_id_from_str(raw: str) -> UserId:
    from alina_rag.domain.models import BotPlatform

    platform_str, _, platform_user_id = raw.partition(":")
    return UserId(platform=BotPlatform(platform_str), platform_user_id=platform_user_id)
