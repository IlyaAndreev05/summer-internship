import logging
import pickle
import re
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

from alina_rag.config import settings
from alina_rag.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


class BM25Store:
    def __init__(self, persist_path: str = "data/bm25_index.pkl"):
        self._chunks: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._persist_path = Path(persist_path)
        self._tokenized: list[list[str]] = []
        if self._persist_path.exists():
            self._load()

    def _load(self):
        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data.get("chunks", [])
            self._metadatas = data.get("metadatas", [])
            self._tokenized = [_tokenize(c) for c in self._chunks]
            if self._tokenized:
                self._bm25 = BM25Okapi(self._tokenized)
        except Exception:
            logger.warning("Failed to load BM25 index, starting fresh")

    def _save(self):
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "metadatas": self._metadatas}, f)

    def add_chunks(self, chunks: list[str], metadatas: list[dict]):
        self._chunks.extend(chunks)
        self._metadatas.extend(metadatas)
        self._tokenized.extend([_tokenize(c) for c in chunks])
        self._bm25 = BM25Okapi(self._tokenized)
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[Document]:
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
                        metadata=self._metadatas[idx] if idx < len(self._metadatas) else {},
                    )
                )
        return results

    def count(self) -> int:
        return len(self._chunks)

    def remove_by_source(self, source: str) -> int:
        """Remove all chunks whose metadata['source'] matches. Returns count removed."""
        keep = [
            i for i, m in enumerate(self._metadatas)
            if m.get("source") != source
        ]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._tokenized = [self._tokenized[i] for i in keep]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._save()
        return removed


def _build_qdrant() -> QdrantVectorStore:
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
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_host,
        temperature=0.1,
    )


def _build_bm25() -> BM25Store:
    return BM25Store()


class RAGAgent:
    def __init__(self):
        self._llm = _build_llm()
        self._vector_store = _build_qdrant()
        self._bm25 = _build_bm25()

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        step_callback: Callable | None = None,
    ) -> str:
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
        return self._vector_store

    def get_bm25_store(self) -> BM25Store:
        return self._bm25

    def remove_by_source(self, source: str) -> None:
        """Remove all chunks for a given source from both Qdrant and BM25."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._vector_store.client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
            ),
        )
        self._bm25.remove_by_source(source)
        logger.info("Removed chunks for source: %s", source)

    def get_llm(self) -> ChatOllama:
        return self._llm
