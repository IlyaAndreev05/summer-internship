"""Base class for LLM providers with auto-download support."""

import logging
from abc import abstractmethod

from alina_rag.domain.interfaces import LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMProvider(LLMProvider):
    """Base LLM provider with model lifecycle management.

    Subclasses implement _download_model() for provider-specific download logic.
    The model is downloaded lazily on first use via _ensure_model().
    """

    def __init__(self) -> None:
        self._ready = False

    async def _ensure_model(self) -> None:
        """Download/pull the model if not already present. Idempotent."""
        if self._ready:
            return
        logger.info("Ensuring LLM model is available...")
        await self._download_model()
        self._ready = True
        logger.info("LLM model ready.")

    @abstractmethod
    async def _download_model(self) -> None:
        """Provider-specific model download. Called once on first use."""
        ...
