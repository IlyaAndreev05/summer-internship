from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from alina_rag.db import Database
from alina_rag.loaders import load_projects
from alina_rag.models import ProjectRecord, ScoredResult, SearchResult

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def rrf_fusion(result_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc
    ranked = sorted(scores, key=lambda sid: scores[sid], reverse=True)
    return [docs[doc_id] for doc_id in ranked]


class TextLookup:
    """Разрешение (source, chunk_index) → текст чанка из кеша БД."""

    def __init__(self, db: Database):
        self._db = db
        self._texts: dict[tuple[str, int], str] = {}
        self.reload()

    def get(self, source: str, chunk_index: int) -> str:
        return self._texts.get((source, chunk_index), "")

    def reload(self) -> None:
        rows = self._db.load_all_chunks()
        self._texts = {(r.source, r.chunk_index): r.chunk_text for r in rows}


class SearchMethod(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        ...


class VectorSearch(SearchMethod):
    """Векторный поиск через Qdrant."""

    def __init__(self, client: QdrantClient, collection: str, embeddings: OllamaEmbeddings):
        self._client = client
        self._collection = collection
        self._embeddings = embeddings

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        try:
            vector = self._embeddings.embed_query(query)
            results = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
            )
        except Exception:
            logger.exception("Vector search failed")
            return []
        out: list[SearchResult] = []
        for r in results:
            p = r.payload or {}
            source = p.get("source", "")
            chunk_index = p.get("chunk_index", 0)
            out.append(SearchResult(
                id=f"{source}:{chunk_index}",
                score=r.score,
                source=source,
                chunk_index=chunk_index,
            ))
        return out


class TrigramSearch(SearchMethod):
    """Триграммный поиск через PostgreSQL."""

    def __init__(self, db: Database):
        self._db = db

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        rows = self._db.trigram_search(query, top_k)
        return [
            SearchResult(
                id=f"{r.source}:{r.chunk_index}",
                score=max(0.0, 1.0 - i * 0.05),
                source=r.source,
                chunk_index=r.chunk_index,
            )
            for i, r in enumerate(rows)
        ]


class BM25Search(SearchMethod):
    """Поиск по BM25Okapi."""

    def __init__(self, db: Database):
        self._db = db
        self._model: BM25Okapi | None = None
        self._metadatas: list[tuple[str, int]] = []
        self._chunk_count: int = -1

    def _ensure_index(self) -> None:
        current_count = len(self._db.load_all_chunks())
        if self._model is not None and self._chunk_count == current_count:
            return
        rows = self._db.load_all_chunks()
        chunks = [r.chunk_text for r in rows]
        self._metadatas = [(r.source, r.chunk_index) for r in rows]
        tokenized = [tokenize(c) for c in chunks]
        self._model = BM25Okapi(tokenized) if tokenized else None
        self._chunk_count = current_count

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        self._ensure_index()
        if not self._model:
            return []
        tokenized = tokenize(query)
        scores = self._model.get_scores(tokenized)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(
                id=f"{self._metadatas[idx][0]}:{self._metadatas[idx][1]}",
                score=score,
                source=self._metadatas[idx][0],
                chunk_index=self._metadatas[idx][1],
            )
            for idx, score in ranked
            if score > 0
        ]


class HybridSearch:
    """Гибридный поиск: Vector + Trigram + BM25 через RRF. Возвращает ScoredResult с текстом."""

    def __init__(self, methods: list[SearchMethod], text_lookup: TextLookup):
        self._methods = methods
        self._texts = text_lookup

    def search(self, query: str, top_k: int = 5) -> list[ScoredResult]:
        all_results: list[list[SearchResult]] = []
        for method in self._methods:
            try:
                all_results.append(method.search(query, top_k=10))
            except Exception:
                logger.exception("Search method %s failed", type(method).__name__)
        if not all_results or not any(all_results):
            return []
        fused = rrf_fusion(all_results)
        top = fused[:top_k]
        results: list[ScoredResult] = []
        for doc in top:
            text = self._texts.get(doc.source, doc.chunk_index)
            if not text:
                text = "(текст не найден)"
            results.append(ScoredResult(
                id=doc.id,
                score=doc.score,
                source=doc.source,
                chunk_index=doc.chunk_index,
                text=text,
            ))
        return results

    def search_results(self, query: str, top_k: int = 5) -> list[ScoredResult]:
        return self.search(query, top_k)


class ProjectSearch:
    """Поиск по таблицам проектов по подстроке."""

    def __init__(self, projects_path: Path):
        self._projects_path = projects_path
        self._cache: list[ProjectRecord] | None = None

    def _ensure_cache(self) -> None:
        if self._cache is not None:
            return
        if not self._projects_path.exists():
            self._cache = []
            return
        self._cache = load_projects(self._projects_path)

    def search(self, query: str, top_k: int = 5) -> list[ProjectRecord]:
        self._ensure_cache()
        if not self._cache:
            return []
        ql = query.lower()
        matches: list[ProjectRecord] = []
        for rec in self._cache:
            if ql in rec.name.lower() or ql in rec.description.lower():
                matches.append(rec)
        return matches[:top_k]

    def search_results(self, query: str, top_k: int = 5) -> list[ScoredResult]:
        records = self.search(query, top_k)
        return [
            ScoredResult(
                id=f"project:{rec.file}:{i}",
                score=1.0 - i * 0.05,
                source=rec.file,
                chunk_index=i,
                text=f"Проект: {rec.name}\nОписание: {rec.description}",
            )
            for i, rec in enumerate(records)
        ]
