"""Sentence-transformers embedding provider with HuggingFace auto-download."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from alina_rag.config import settings
from alina_rag.infrastructure.base_embedding import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class STEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using SentenceTransformers.

    Model is auto-downloaded from HuggingFace on first use.
    To switch to a different embedding backend, subclass BaseEmbeddingProvider.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: SentenceTransformer | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def _download_model(self) -> None:
        """Download the SentenceTransformer model from HuggingFace."""
        logger.info("Loading embedding model %s (first use — may download)...", settings.embed_model)
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            self._executor,
            lambda: SentenceTransformer(
                settings.embed_model,
                device=settings.embed_device,
            ),
        )
        logger.info("Embedding model %s loaded.", settings.embed_model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors."""
        await self._ensure_model()
        assert self._model is not None
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            self._executor, self._model.encode, texts
        )
        return [emb.tolist() for emb in embeddings]
