"""Domain entities and value objects."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


# ── Value Objects ────────────────────────────────────


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DocSource(str, Enum):
    MANUAL = "manual"
    EXAMPLE_LIBRARY = "example_library"


class BotPlatform(str, Enum):
    CONSOLE = "console"
    TELEGRAM = "telegram"
    VK = "vk"
    API = "api"


@dataclass(frozen=True)
class UserId:
    """Composite user identity: platform + platform-specific id."""
    platform: BotPlatform
    platform_user_id: str

    def __str__(self) -> str:
        return f"{self.platform.value}:{self.platform_user_id}"


# ── Entities ─────────────────────────────────────────


@dataclass
class Message:
    id: str = field(default_factory=lambda: uuid4().hex)
    user_id: UserId | None = None
    role: Role = Role.USER
    content: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Document:
    """A document ingested into the knowledge base."""
    id: str = field(default_factory=lambda: uuid4().hex)
    source: DocSource = DocSource.MANUAL
    filename: str = ""
    title: str = ""
    content: str = ""
    chunk_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class AgentStep:
    """One step in the ReAct agent's reasoning loop."""
    thought: str = ""
    action: str = ""          # "search" | "answer" | "clarify"
    action_input: str = ""    # search query, or empty for direct answer
    observation: str = ""     # result from the tool


@dataclass
class ChatSession:
    """A user's chat session with message history."""
    user_id: UserId
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 20

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self._trim()

    def _trim(self) -> None:
        """Keep only the last N messages."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    @property
    def history_text(self) -> str:
        """Format message history for the LLM context."""
        lines: list[str] = []
        for msg in self.messages[:-1]:  # all except the latest (which is current query)
            role_label = "Пользователь" if msg.role == Role.USER else "Ассистент"
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)
