"""Abstract interfaces (ports) for the domain layer."""

from abc import ABC, abstractmethod

from alina_rag.domain.models import (
    ChatSession,
    Document,
    Message,
    SearchResult,
    UserId,
)


class ChatRepository(ABC):
    """Persistent storage for chat history."""

    @abstractmethod
    async def save_message(self, message: Message) -> None:
        ...

    @abstractmethod
    async def get_history(
        self, user_id: UserId, limit: int = 20
    ) -> list[Message]:
        ...

    @abstractmethod
    async def clear_history(self, user_id: UserId) -> None:
        ...


class DocumentRepository(ABC):
    """Storage for ingested documents and their chunks."""

    @abstractmethod
    async def add_document(self, doc: Document) -> None:
        ...

    @abstractmethod
    async def list_documents(self) -> list[Document]:
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        ...


class VectorStore(ABC):
    """Vector store for semantic search over document chunks."""

    @abstractmethod
    async def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        metadata: list[dict[str, str]],
    ) -> None:
        ...

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 5
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...


class EmbeddingProvider(ABC):
    """Text → embedding vector."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        ...


class LLMProvider(ABC):
    """Large language model provider (local or remote)."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts."""
        ...

    @abstractmethod
    async def generate_with_history(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        """Generate a response with conversation history."""
        ...


class SessionStore(ABC):
    """In-memory or persistent store for active chat sessions."""

    @abstractmethod
    def get_or_create(self, user_id: UserId, max_messages: int = 20) -> ChatSession:
        ...

    @abstractmethod
    def remove(self, user_id: UserId) -> None:
        ...
