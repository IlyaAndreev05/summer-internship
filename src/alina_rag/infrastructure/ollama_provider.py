"""Ollama LLM provider with auto-pull support."""

import logging

from ollama import AsyncClient

from alina_rag.config import settings
from alina_rag.domain.models import Message
from alina_rag.infrastructure.base_llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a local Ollama instance.

    Model is auto-pulled on first use if not already present.
    To switch to a different LLM backend, subclass BaseLLMProvider.
    """

    def __init__(self) -> None:
        super().__init__()
        self._client = AsyncClient(host=settings.ollama_host)
        self._model = settings.llm_model

    async def _download_model(self) -> None:
        """Pull the Ollama model if not already available locally."""
        logger.info("Checking Ollama model %s...", self._model)
        try:
            models = await self._client.list()
        except Exception:
            logger.warning("Cannot reach Ollama at %s — assuming model is available.", settings.ollama_host)
            return

        base_name = self._model.split(":")[0]
        for m in models.get("models", []):
            if m.get("name", "").startswith(base_name):
                logger.info("Model %s already present.", self._model)
                return

        logger.info("Pulling model %s (this may take a while)...", self._model)
        await self._client.pull(self._model)
        logger.info("Model %s pulled successfully.", self._model)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from system and user prompts."""
        await self._ensure_model()
        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]

    async def generate_with_history(
        self, system_prompt: str, messages: list[Message]
    ) -> str:
        """Generate a response incorporating conversation history."""
        await self._ensure_model()
        ollama_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in messages:
            ollama_messages.append(
                {"role": msg.role.value, "content": msg.content}
            )
        response = await self._client.chat(
            model=self._model,
            messages=ollama_messages,
        )
        return response["message"]["content"]
