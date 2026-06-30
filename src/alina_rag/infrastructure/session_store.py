"""In-memory session store implementation."""

from alina_rag.config import settings
from alina_rag.domain.interfaces import SessionStore
from alina_rag.domain.models import ChatSession, UserId


class InMemorySessionStore(SessionStore):
    """Session store backed by an in-memory dictionary."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, user_id: UserId, max_messages: int | None = None) -> ChatSession:
        """Return the existing session for ``user_id`` or create a new one."""
        key = str(user_id)
        session = self._sessions.get(key)
        if session is None:
            session = ChatSession(
                user_id=user_id,
                max_messages=max_messages if max_messages is not None else settings.chat_max_messages,
            )
            self._sessions[key] = session
        return session

    def remove(self, user_id: UserId) -> None:
        """Remove the session for ``user_id`` if it exists."""
        self._sessions.pop(str(user_id), None)
