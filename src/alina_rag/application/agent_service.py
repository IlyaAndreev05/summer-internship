"""ReAct agent — core reasoning loop for the ALINA GPSS AI Consultant."""

import re
from typing import TYPE_CHECKING, Callable

from alina_rag.config import settings
from alina_rag.application.prompts import build_system_prompt
from alina_rag.domain.models import AgentStep, Message, Role, UserId

if TYPE_CHECKING:
    from alina_rag.domain.interfaces import LLMProvider, SessionStore, VectorStore
    from alina_rag.infrastructure.bm25_store import BM25Store
#
# System prompt is loaded dynamically via alina_rag.application.prompts.build_system_prompt()
# Override via AGENT_SYSTEM_PROMPT and AGENT_EXTRA_CONTEXT env vars or .env file.


SEARCH_DOCS_PATTERN = re.compile(r'search_documents\("(.+?)"\)', re.DOTALL)
SEARCH_KW_PATTERN = re.compile(r'search_keywords\("(.+?)"\)', re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)
MAX_REACT_STEPS = 5
MAX_SEARCHES = 3

class AgentService:
    """ReAct agent that answers user questions using dual search (vector + BM25)."""

    def __init__(
        self,
        llm: "LLMProvider",
        vector_store: "VectorStore",
        bm25_store: "BM25Store | None" = None,
        session_store: "SessionStore | None" = None,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._session_store = session_store

    async def process_message(
        self, user_id: UserId, message_text: str,
        step_callback: "Callable[[AgentStep], None] | None" = None,
    ) -> str:
        """Process a user message and return the assistant's response."""
        session = self._session_store.get_or_create(
            user_id, max_messages=settings.chat_max_messages
        )

        user_message = Message(user_id=user_id, role=Role.USER, content=message_text)
        session.add_message(user_message)

        history_text = session.history_text
        response = await self._react_loop(message_text, history_text, step_callback)

        assistant_message = Message(user_id=user_id, role=Role.ASSISTANT, content=response)
        session.add_message(assistant_message)

        return response

    async def _react_loop(
        self, query: str, history: str,
        step_callback: "Callable[[AgentStep], None] | None" = None,
    ) -> str:
        """Run the ReAct reasoning loop (max MAX_REACT_STEPS steps)."""
        steps: list[AgentStep] = []
        search_count = 0

        for _ in range(MAX_REACT_STEPS):
            context = self._build_context(query, history, steps)
            llm_output = await self._llm.generate(build_system_prompt(), context)

            final_answer = self._extract_final_answer(llm_output)
            if final_answer is not None:
                return final_answer

            search_query, tool = self._extract_any_search(llm_output)
            if search_query is not None and search_count < MAX_SEARCHES:
                search_count += 1
                if tool == "keywords" and self._bm25_store is not None:
                    observation = self._execute_keyword_search(search_query)
                else:
                    observation = await self._execute_search(search_query)
                step = self._parse_step(llm_output, observation)
                steps.append(step)
                if step_callback:
                    step_callback(step)
                continue

            # No search and no final answer — store step and continue
            step = self._parse_step(llm_output, "")
            steps.append(step)
            if step_callback:
                step_callback(step)

        # Fallback: return last meaningful output
        return self._fallback_answer(steps)

    # ── Context building ─────────────────────────────────────────────

    def _build_context(self, query: str, history: str, steps: list[AgentStep]) -> str:
        """Build the user prompt for the current ReAct iteration."""
        parts: list[str] = []

        if history:
            parts.append(f"История диалога:\n{history}\n")

        parts.append(f"Вопрос пользователя: {query}\n")

        if steps:
            parts.append("Предыдущие шаги:")
            for i, step in enumerate(steps, 1):
                parts.append(f"\nШаг {i}:")
                if step.thought:
                    parts.append(f"Thought: {step.thought}")
                if step.action_input:
                    parts.append(f'Action: search_documents("{step.action_input}")')
                if step.observation:
                    parts.append(f"Observation: {step.observation}")

        parts.append("\nПродолжи рассуждение. Используй формат Thought/Action или Final Answer.")

        return "\n".join(parts)

    # ── Parsing ──────────────────────────────────────────────────────

    def _extract_any_search(self, text: str) -> tuple[str | None, str | None]:
        """Extract search query and tool type from LLM output.

        Returns:
            (query, tool) where tool is "documents" or "keywords", or (None, None).
        """
        # Try semantic search first
        match = SEARCH_DOCS_PATTERN.search(text)
        if match:
            return match.group(1).strip(), "documents"

        # Try keyword search
        match = SEARCH_KW_PATTERN.search(text)
        if match:
            return match.group(1).strip(), "keywords"

        return None, None

    def _extract_final_answer(self, text: str) -> str | None:
        """Extract Final Answer from LLM output."""
        match = FINAL_ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _parse_step(self, llm_output: str, observation: str) -> AgentStep:
        """Parse an LLM output line into an AgentStep."""
        thought = ""
        action_input = ""
        action = "answer"

        for line in llm_output.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("thought:"):
                thought = stripped.split(":", 1)[1].strip()
            elif "search_documents(" in stripped:
                m = SEARCH_DOCS_PATTERN.search(stripped)
                if m:
                    action_input = m.group(1).strip()
                    action = "search"
            elif "search_keywords(" in stripped:
                m = SEARCH_KW_PATTERN.search(stripped)
                if m:
                    action_input = m.group(1).strip()
                    action = "search"

        return AgentStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )

    # ── Tool execution ───────────────────────────────────────────────

    async def _execute_search(self, query: str) -> str:
        """Execute a vector store search and format results."""
        results = await self._vector_store.search(query, top_k=3)
        if not results:
            return "Результатов не найдено (семантический поиск)."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] (релевантность: {r.score:.2f})\n{r.content}")
        return "\n\n".join(lines)

    def _execute_keyword_search(self, query: str) -> str:
        """Execute a BM25 keyword search and format results."""
        if self._bm25_store is None:
            return "Поиск по ключевым словам недоступен."

        results = self._bm25_store.search(query, top_k=3)
        if not results:
            return "Результатов не найдено (поиск по ключевым словам)."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] (BM25: {r.score:.2f})\n{r.content}")
        return "\n\n".join(lines)

    # ── Fallback ─────────────────────────────────────────────────────

    def _fallback_answer(self, steps: list[AgentStep]) -> str:
        """Generate a fallback when the agent doesn't produce a Final Answer."""
        if steps:
            last = steps[-1]
            if last.thought:
                return (
                    f"Извините, я не смог сформулировать точный ответ. "
                    f"Вот что я думаю: {last.thought}"
                )
        return (
            "Извините, мне не удалось найти ответ на ваш вопрос. "
            "Пожалуйста, переформулируйте его или задайте другой вопрос по GPSS."
        )
