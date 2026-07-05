import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

from alina_rag.config import settings
from alina_rag.db import load_all_chunks, trigram_search
from alina_rag.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


def _tokenize(text: str) -> list[str]:
    """Токенизация текста: нижний регистр, кириллица и латиница с цифрами."""
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


_bm25_cache: dict[str, Any] = {}


def _get_bm25() -> tuple[list[str], list[dict], BM25Okapi | None]:
    """Возвращает чанки, метаданные и модель BM25, строит индекс при первом вызове."""
    if "model" not in _bm25_cache:
        rows = load_all_chunks()
        chunks = [r[3] for r in rows]
        metadatas = [{"source": r[1], "filename": r[2]} for r in rows]
        tokenized = [_tokenize(c) for c in chunks]
        model = BM25Okapi(tokenized) if tokenized else None
        _bm25_cache["chunks"] = chunks
        _bm25_cache["metadatas"] = metadatas
        _bm25_cache["model"] = model
        logger.info("BM25 index built from %d chunks", len(chunks))
    return _bm25_cache["chunks"], _bm25_cache["metadatas"], _bm25_cache["model"]


def bm25_search(query: str, top_k: int = 5) -> list[Document]:
    """Ключевой поиск по BM25."""
    chunks, metadatas, model = _get_bm25()
    if not model:
        return []
    tokenized = _tokenize(query)
    scores = model.get_scores(tokenized)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results: list[Document] = []
    for idx, score in ranked:
        if score > 0:
            results.append(
                Document(page_content=chunks[idx], metadata=metadatas[idx])
            )
    return results


def _build_qdrant() -> QdrantVectorStore:
    """Создаёт и настраивает векторное хранилище Qdrant."""
    client = QdrantClient(url=settings.qdrant_url)
    embeddings = OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_host,
    )
    try:
        client.get_collection(settings.qdrant_collection)
    except Exception:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def _build_llm() -> ChatOllama:
    """Создаёт LLM-клиент Ollama."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_host,
        temperature=0.1,
    )


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_vector",
            "description": "Семантический векторный поиск. Хорош для общих и концептуальных вопросов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_keywords",
            "description": "Нечёткий поиск по триграммам. Хорош для точных терминов и названий блоков.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_bm25",
            "description": "BM25 поиск по словам. Хорош для многокомпонентных запросов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_ACTION_RE = re.compile(
    r'Action:\s*(search_vector|search_keywords|search_bm25)\s*\(\s*["\'](.+?)["\']\s*\)',
    re.IGNORECASE,
)


class RAGAgent:
    """ReAct RAG-агент: LLM решает когда и как искать."""

    def __init__(self):
        """Инициализирует LLM и векторное хранилище. BM25 строится лениво."""
        self._llm = _build_llm()
        self._vector_store = _build_qdrant()

    def _search_vector(self, query: str) -> str:
        """Семантический поиск по векторам."""
        results = self._vector_store.similarity_search(query, k=5)
        if not results:
            return "Ничего не найдено."
        parts = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (Источник: {src})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _search_keywords(self, query: str) -> str:
        """Нечёткий поиск по триграммам."""
        rows = trigram_search(query, top_k=5)
        if not rows:
            return "Ничего не найдено."
        parts = []
        for i, row in enumerate(rows, 1):
            _, source, _filename, text, _ = row
            parts.append(f"[{i}] (Источник: {source})\n{text}")
        return "\n\n".join(parts)

    def _search_bm25(self, query: str) -> str:
        """Ключевой поиск по BM25."""
        results = bm25_search(query, top_k=5)
        if not results:
            return "Ничего не найдено."
        parts = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (Источник: {src})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _execute_tool(self, name: str, query: str) -> str:
        """Вызов инструмента по имени."""
        dispatch = {
            "search_vector": self._search_vector,
            "search_keywords": self._search_keywords,
            "search_bm25": self._search_bm25,
        }
        fn = dispatch.get(name)
        if fn is None:
            return f"Неизвестный инструмент: {name}"
        return fn(query)

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        """ReAct-цикл: LLM решает когда искать, выполняет инструменты, возвращает ответ."""
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
                    tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                    args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        import json
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    query_val = args.get("query", "")
                    logger.info("Tool call: %s(%s)", tool_name, query_val)

                    result = self._execute_tool(tool_name, query_val)

                    if step_callback:
                        step_callback({
                            "type": "tool_call",
                            "tool": tool_name,
                            "query": query_val,
                            "iteration": iteration,
                        })

                    tool_call_id = tc.get("id", "")
                    messages.append(ToolMessage(
                        content=result,
                        tool_call_id=tool_call_id,
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

    def get_vector_store(self) -> QdrantVectorStore:
        """Возвращает векторное хранилище Qdrant."""
        return self._vector_store
    def get_llm(self) -> ChatOllama:
        """Возвращает экземпляр LLM (для test_mode)."""
        return self._llm
