"""Base class for embedding providers with auto-download support."""

import logging
from abc import abstractmethod

from alina_rag.domain.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(EmbeddingProvider):
    """Base embedding provider with model lifecycle management.

    Subclasses implement _download_model() for provider-specific download logic.
    The model is downloaded lazily on first use via _ensure_model().
    """

    def __init__(self) -> None:
        self._ready = False

    async def _ensure_model(self) -> None:
        """Download the model if not already present. Idempotent."""
        if self._ready:
            return
        logger.info("Downloading embedding model (if needed)...")
        await self._download_model()
        self._ready = True
        logger.info("Embedding model ready.")

    @abstractmethod
    async def _download_model(self) -> None:
        """Provider-specific model download. Called once on first use."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Encode a single query string."""
        results = await self.embed([query])
        return results[0]
