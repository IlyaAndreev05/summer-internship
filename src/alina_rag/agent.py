import logging
import re
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

from alina_rag.config import settings
from alina_rag.db import load_all_chunks
from alina_rag.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Токенизация текста: нижний регистр, кириллица и латиница с цифрами."""
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


class BM25Search:
    """In-memory BM25 index built from Postgres chunks on startup."""

    def __init__(self):
        """Инициализирует BM25-индекс из чанков в базе."""
        rows = load_all_chunks()  # (id, source, filename, chunk_text, chunk_index)
        self._chunks: list[str] = [r[3] for r in rows]
        self._metadatas: list[dict] = [{"source": r[1], "filename": r[2]} for r in rows]
        tokenized = [_tokenize(c) for c in self._chunks]
        self._bm25: BM25Okapi | None = BM25Okapi(tokenized) if tokenized else None
        logger.info("BM25 index built from %d chunks", len(self._chunks))

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Поиск по BM25 с возвратом top_k документов."""
        if not self._bm25:
            return []
        tokenized = _tokenize(query)
        scores = self._bm25.get_scores(tokenized)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[Document] = []
        for idx, score in ranked:
            if score > 0:
                results.append(
                    Document(
                        page_content=self._chunks[idx],
                        metadata=self._metadatas[idx],
                    )
                )
        return results

    def count(self) -> int:
        """Количество чанков в индексе."""
        return len(self._chunks)


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


class RAGAgent:
    """RAG-агент: векторный + BM25 поиск и генерация ответов."""
    def __init__(self):
        """Инициализирует LLM, векторное хранилище и BM25-индекс."""
        self._llm = _build_llm()
        self._vector_store = _build_qdrant()
        self._bm25 = BM25Search()

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
        """Генерирует ответ на вопрос с учётом истории и найденного контекста."""
        vector_results = self._vector_store.similarity_search(question, k=5)
        bm25_results = self._bm25.search(question, top_k=5)

        seen = set()
        merged: list[Document] = []
        for doc in vector_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        for doc in bm25_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        context_parts: list[str] = []
        for i, doc in enumerate(merged[:8], 1):
            src = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
            context_parts.append(f"[{i}] (Источник: {src})\n{doc.page_content}")

        context = "\n\n".join(context_parts) if context_parts else "Информация не найдена."

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            messages.extend(history)

        user_prompt = (
            f"Контекст из документации:\n\n{context}\n\n"
            f"Вопрос пользователя: {question}\n\n"
            f"Ответь на вопрос на основе контекста. Если в контексте нет информации для ответа, "
            f"скажи об этом честно. Если информация найдена, дай развернутый ответ."
        )
        messages.append({"role": "user", "content": user_prompt})

        if step_callback:
            step_callback(
                {
                    "type": "search",
                    "vector_count": len(vector_results),
                    "bm25_count": len(bm25_results),
                }
            )

        response = self._llm.invoke(messages)
        return response.content

    def get_vector_store(self) -> QdrantVectorStore:
        """Возвращает векторное хранилище Qdrant."""
        return self._vector_store

    def get_llm(self) -> ChatOllama:
        """Возвращает экземпляр LLM."""
        return self._llm
