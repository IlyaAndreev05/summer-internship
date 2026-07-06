import logging
import re
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from alina_rag.config import settings
from alina_rag.prompts import AUTO_RAG_PROMPT, SYSTEM_PROMPT
from alina_rag.search import BM25Search, HybridSearch, ProjectSearch, TrigramSearch, VectorSearch

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Поиск по документации GPSS — ищет по смыслу, ключевым словам и точным терминам одновременно.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_projects",
            "description": "Поиск проектов по названию и описанию из data/projects.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
                "required": ["query"],
            },
        },
    },
]

_ACTION_RE = re.compile(
    r'Action:\s*(search_docs|search_projects)\s*\(\s*["\'](.+?)["\']\s*\)',
    re.IGNORECASE,
)


class RAGAgent:
    def __init__(self):
        self._llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_host,
            temperature=0.1,
        )
        qdrant = QdrantClient(url=settings.qdrant_url)
        embeddings = OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_host,
        )
        try:
            qdrant.get_collection(settings.qdrant_collection)
        except Exception:
            qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        self._docs = HybridSearch([
            VectorSearch(qdrant, settings.qdrant_collection, embeddings),
            TrigramSearch(),
            BM25Search(),
        ])
        self._projects = ProjectSearch()
        self._qdrant = qdrant
        self._embeddings = embeddings

    def get_llm(self):
        return self._llm

    def _execute_tool(self, name: str, query: str) -> str:
        if name == "search_docs":
            return self._docs.search(query)
        if name == "search_projects":
            return self._projects.search(query)
        return f"Неизвестный инструмент: {name}"

    def _extract_tool_args(self, tool_call: dict) -> tuple[str, str]:
        name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
        args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return name, args.get("query", "")

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        if settings.rag_mode == "auto":
            return self._answer_auto(question, history, step_callback)
        return self._answer_tools(question, history, step_callback)

    def _answer_auto(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        if step_callback:
            step_callback({"type": "search", "query": question, "iteration": 0})

        docs = self._docs.search(question)
        projects = self._projects.search(question)
        context = f"Документация:\n{docs}\n\nПроекты:\n{projects}"

        messages: list = [SystemMessage(content=AUTO_RAG_PROMPT)]
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=f"Контекст:\n{context}\n\nВопрос: {question}"))

        response = self._llm.invoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)

    def _answer_tools(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

        if history:
            for msg in history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=question))

        for iteration in range(MAX_ITERATIONS):
            try:
                llm_with_tools = self._llm.bind_tools(TOOLS_SCHEMA)
                response: AIMessage = llm_with_tools.invoke(messages)
            except Exception:
                response = self._llm.invoke(messages)

            if hasattr(response, "tool_calls") and response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    tool_name, query_val = self._extract_tool_args(tc)
                    logger.info("Tool call: %s(%s)", tool_name, query_val)

                    result = self._execute_tool(tool_name, query_val)

                    if step_callback:
                        step_callback({
                            "type": "tool_call",
                            "tool": tool_name,
                            "query": query_val,
                            "iteration": iteration,
                        })

                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tc.get("id", ""),
                    ))
                continue

            content = response.content if isinstance(response.content, str) else str(response.content)
            match = _ACTION_RE.search(content)
            if match:
                tool_name = match.group(1)
                query_val = match.group(2)
                logger.info("Parsed action: %s(%s)", tool_name, query_val)

                result = self._execute_tool(tool_name, query_val)

                if step_callback:
                    step_callback({
                        "type": "tool_call",
                        "tool": tool_name,
                        "query": query_val,
                        "iteration": iteration,
                    })

                truncated = content[: match.start()].rstrip()
                if truncated:
                    messages.append(AIMessage(content=truncated))

                tool_msg = (
                    f"Результат поиска {tool_name}(\"{query_val}\"):\n\n{result}\n\n"
                    "Проанализируй результат и дай краткий ответ без нумерации [1] и без указания источников."
                )
                messages.append(HumanMessage(content=tool_msg))
                continue

            return content

        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
