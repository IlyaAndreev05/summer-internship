"""Chat orchestration — saves messages and delegates to the agent."""

from typing import TYPE_CHECKING

from alina_rag.domain.models import Message, Role, UserId

if TYPE_CHECKING:
    from alina_rag.application.agent_service import AgentService
    from alina_rag.domain.interfaces import ChatRepository


class ChatService:
    """Orchestrates a chat turn: persist messages, run agent, return response."""

    def __init__(self, agent: "AgentService", chat_repo: "ChatRepository") -> None:
        self._agent = agent
        self._chat_repo = chat_repo

    async def handle_message(self, user_id: UserId, message_text: str) -> str:
        """Process a user message through the full pipeline."""
        user_message = Message(user_id=user_id, role=Role.USER, content=message_text)
        await self._chat_repo.save_message(user_message)

        response_text = await self._agent.process_message(user_id, message_text)

        assistant_message = Message(user_id=user_id, role=Role.ASSISTANT, content=response_text)
        await self._chat_repo.save_message(assistant_message)

        return response_text
