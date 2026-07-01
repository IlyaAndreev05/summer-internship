"""OpenAI-compatible LLM provider (remote API, no download needed)."""

from openai import AsyncOpenAI

from alina_rag.config import settings
from alina_rag.domain.models import Message
from alina_rag.infrastructure.base_llm import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by an OpenAI-compatible API endpoint.

    No model download — remote API is always available.
    To switch to a different remote provider, subclass BaseLLMProvider.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__()
        _api_key = api_key or settings.llm_api_key or "not-needed"
        self._client = AsyncOpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=_api_key,
        )
        self._model = model or settings.llm_model

    async def _download_model(self) -> None:
        """No-op: remote models don't need downloading."""
        pass

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from system and user prompts."""
        await self._ensure_model()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def generate_with_history(
        self, system_prompt: str, messages: list[Message]
    ) -> str:
        """Generate a response incorporating conversation history."""
        await self._ensure_model()
        openai_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in messages:
            openai_messages.append(
                {"role": msg.role.value, "content": msg.content}
            )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
        )
        return response.choices[0].message.content or ""
