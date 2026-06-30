"""BM25 keyword-based search store for exact term matching."""

import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

from alina_rag.config import settings
from alina_rag.domain.models import SearchResult

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple Russian-aware tokenizer: lowercase, split on whitespace/punctuation."""
    import re
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


class BM25Store:
    """In-memory BM25 index for keyword search.

    Complements the vector store — catches exact term matches
    that semantic search might miss.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._chunks: list[str] = []
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._metadatas: list[dict[str, str]] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._persist_path = persist_path

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        metadata: list[dict[str, str]],
    ) -> None:
        """Add document chunks to the BM25 index."""
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            self._chunks.append(chunk)
            self._chunk_ids.append(chunk_id)
            self._doc_ids.append(doc_id)
            meta = metadata[i] if i < len(metadata) else {}
            self._metadatas.append(meta)
            self._tokenized.append(_tokenize(chunk))

        # Rebuild BM25 index
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

        logger.debug("BM25 index: %d chunks total", len(self._chunks))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the BM25 index with a keyword query."""
        if self._bm25 is None or not self._tokenized:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        # Get top-k indices
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in indexed_scores[:top_k]:
            if score <= 0:
                continue
            # Normalize score to 0..1 range (rough)
            max_score = max(scores) if max(scores) > 0 else 1.0
            normalized = score / max_score
            results.append(
                SearchResult(
                    chunk_id=self._chunk_ids[idx],
                    document_id=self._doc_ids[idx],
                    content=self._chunks[idx],
                    metadata=self._metadatas[idx],
                    score=normalized,
                )
            )
        return results

    def count(self) -> int:
        """Total chunks in the BM25 index."""
        return len(self._chunks)

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks for a document from the index."""
        indices_to_keep = [
            i for i, did in enumerate(self._doc_ids) if did != doc_id
        ]
        self._chunks = [self._chunks[i] for i in indices_to_keep]
        self._chunk_ids = [self._chunk_ids[i] for i in indices_to_keep]
        self._doc_ids = [self._doc_ids[i] for i in indices_to_keep]
        self._metadatas = [self._metadatas[i] for i in indices_to_keep]
        self._tokenized = [self._tokenized[i] for i in indices_to_keep]

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None
