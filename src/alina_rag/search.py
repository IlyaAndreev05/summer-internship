from abc import ABC, abstractmethod
import logging
import re
from typing import Any

import pandas as pd
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from alina_rag.config import settings
from alina_rag.db import load_all_chunks, trigram_search

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def rrf_fusion(results_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [docs[doc_id] for doc_id in ranked]


class TextLookup:
    def __init__(self):
        rows = load_all_chunks()
        self._texts: dict[tuple[str, int], str] = {
            (r[1], r[4]): r[3] for r in rows
        }

    def get(self, source: str, chunk_index: int) -> str:
        return self._texts.get((source, chunk_index), "")

    def reload(self):
        rows = load_all_chunks()
        self._texts = {(r[1], r[4]): r[3] for r in rows}


class SearchMethod(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        ...


class VectorSearch(SearchMethod):
    def __init__(self, client: QdrantClient, collection: str, embeddings: OllamaEmbeddings):
        self._client = client
        self._collection = collection
        self._embeddings = embeddings

    def search(self, query: str, top_k: int = 10) -> list[dict]:
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
        out = []
        for r in results:
            p = r.payload or {}
            out.append({
                "id": f"{p.get('source', '')}:{p.get('chunk_index', 0)}",
                "score": r.score,
                "source": p.get("source", ""),
                "chunk_index": p.get("chunk_index", 0),
            })
        return out


class TrigramSearch(SearchMethod):
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        rows = trigram_search(query, top_k)
        return [
            {
                "id": f"{r[1]}:{r[4]}",
                "score": max(0.0, 1.0 - i * 0.05),
                "source": r[1],
                "chunk_index": r[4],
            }
            for i, r in enumerate(rows)
        ]


class BM25Search(SearchMethod):
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _ensure_index(self):
        if "model" in self._cache:
            return
        rows = load_all_chunks()
        chunks = [r[3] for r in rows]
        metadatas = [{"source": r[1], "chunk_index": r[4]} for r in rows]
        tokenized = [_tokenize(c) for c in chunks]
        self._cache["model"] = BM25Okapi(tokenized) if tokenized else None
        self._cache["metadatas"] = metadatas

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        self._ensure_index()
        model = self._cache.get("model")
        if not model:
            return []
        tokenized = _tokenize(query)
        scores = model.get_scores(tokenized)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "id": f"{self._cache['metadatas'][idx]['source']}:{self._cache['metadatas'][idx]['chunk_index']}",
                "score": score,
                "source": self._cache["metadatas"][idx]["source"],
                "chunk_index": self._cache["metadatas"][idx]["chunk_index"],
            }
            for idx, score in ranked
            if score > 0
        ]


class HybridSearch:
    def __init__(self, methods: list[SearchMethod]):
        self._methods = methods
        self._texts = TextLookup()

    def search(self, query: str, top_k: int = 5) -> str:
        all_results: list[list[dict]] = []
        for method in self._methods:
            try:
                all_results.append(method.search(query, top_k=10))
            except Exception:
                logger.exception("Search method %s failed", type(method).__name__)
        if not all_results or not any(all_results):
            return "Ничего не найдено."
        fused = rrf_fusion(all_results)
        top = fused[:top_k]
        parts = []
        for i, doc in enumerate(top, 1):
            src = doc.get("source", "")
            ci = doc.get("chunk_index", 0)
            text = self._texts.get(src, ci)
            if not text:
                text = "(текст не найден)"
            parts.append(f"[{i}] (Источник: {src})\n{text}")
        return "\n\n".join(parts)


class ProjectSearch:
    def search(self, query: str, top_k: int = 5) -> str:
        pp = settings.projects_path
        if not pp.exists():
            return "Папка с проектами не найдена."
        rows: list[dict] = []
        for fp in sorted(pp.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in (".xlsx", ".csv"):
                continue
            try:
                df = pd.read_csv(fp) if fp.suffix.lower() == ".csv" else pd.read_excel(fp)
            except Exception:
                logger.exception("Failed to read %s", fp)
                continue
            nc = next((c for c in df.columns if c.lower() == "name"), None)
            dc = next((c for c in df.columns if c.lower() == "description"), None)
            if nc is None and dc is None:
                continue
            for _, row in df.iterrows():
                nv = str(row[nc]) if nc else ""
                dv = str(row[dc]) if dc else ""
                ql = query.lower()
                if ql in nv.lower() or ql in dv.lower():
                    rows.append({"file": fp.name, "name": nv, "description": dv})
        if not rows:
            return "Проекты не найдены."
        parts = []
        for i, r in enumerate(rows[:top_k], 1):
            parts.append(f"[{i}] {r['name']} ({r['file']})\n{r['description']}")
        return "\n\n".join(parts)
