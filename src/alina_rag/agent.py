from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient

from alina_rag.config import Settings
from alina_rag.db import Database
from alina_rag.models import ScoredResult
from alina_rag.prompts import AUTO_RAG_PROMPT, SYSTEM_PROMPT
from alina_rag.search import (
    BM25Search,
    HybridSearch,
    ProjectSearch,
    TextLookup,
    TrigramSearch,
    VectorSearch,
)

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
    """RAG-агент с режимами auto и tools."""

    def __init__(self, db: Database, cfg: Settings) -> None:
        self._cfg = cfg
        self._llm = ChatOllama(
            model=cfg.llm_model,
            base_url=cfg.ollama_host,
            temperature=0.1,
        )
        self._embeddings = OllamaEmbeddings(
            model=cfg.embed_model,
            base_url=cfg.ollama_host,
        )
        qdrant = QdrantClient(url=cfg.qdrant_url)
        text_lookup = TextLookup(db)
        methods = [
            VectorSearch(qdrant, cfg.qdrant_collection, self._embeddings),
            TrigramSearch(db),
            BM25Search(db),
        ]
        self._docs = HybridSearch(methods, text_lookup)
        self._projects = ProjectSearch(cfg.projects_path)

    def get_llm(self):
        return self._llm

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        if self._cfg.rag_mode == "auto":
            return self._answer_auto(question, history, step_callback)
        return self._answer_tools(question, history, step_callback)

    def _format_results(self, results: list[ScoredResult]) -> str:
        if not results:
            return "Ничего не найдено."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] (Источник: {r.source})\n{r.text}")
        return "\n\n".join(parts)

    def _answer_auto(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        docs_results = self._docs.search_results(question)
        projects_results = self._projects.search_results(question)

        if step_callback:
            if self._cfg.chat_verbose:
                step_callback({
                    "type": "search",
                    "query": question,
                    "results": docs_results,
                })
            else:
                step_callback({"type": "search", "query": question})

        docs_text = self._format_results(docs_results)
        projects_text = self._format_results(projects_results)
        context = f"Документация:\n{docs_text}\n\nПроекты:\n{projects_text}"

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

                    results = self._search_results(tool_name, query_val)
                    result_text = self._format_results(results)

                    if step_callback:
                        cb: dict = {
                            "type": "tool_call",
                            "tool": tool_name,
                            "query": query_val,
                            "iteration": iteration,
                        }
                        if self._cfg.chat_verbose:
                            cb["results"] = results
                        step_callback(cb)

                    messages.append(ToolMessage(
                        content=result_text,
                        tool_call_id=tc.get("id", ""),
                    ))
                continue

            content = response.content if isinstance(response.content, str) else str(response.content)
            match = _ACTION_RE.search(content)
            if match:
                tool_name = match.group(1)
                query_val = match.group(2)
                logger.info("Parsed action: %s(%s)", tool_name, query_val)

                results = self._search_results(tool_name, query_val)
                result_text = self._format_results(results)

                if step_callback:
                    cb = {
                        "type": "tool_call",
                        "tool": tool_name,
                        "query": query_val,
                        "iteration": iteration,
                    }
                    if self._cfg.chat_verbose:
                        cb["results"] = results
                    step_callback(cb)

                truncated = content[: match.start()].rstrip()
                if truncated:
                    messages.append(AIMessage(content=truncated))

                tool_msg = (
                    f'Результат поиска {tool_name}("{query_val}"):\n\n{result_text}\n\n'
                    "Проанализируй результат и дай краткий ответ без нумерации [1] и без указания источников."
                )
                messages.append(HumanMessage(content=tool_msg))
                continue

            return content

        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)

    def _search_results(self, name: str, query: str) -> list[ScoredResult]:
        if name == "search_docs":
            return self._docs.search_results(query)
        if name == "search_projects":
            return self._projects.search_results(query)
        return []

    def _extract_tool_args(self, tool_call: dict) -> tuple[str, str]:
        name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
        args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return name, args.get("query", "")
