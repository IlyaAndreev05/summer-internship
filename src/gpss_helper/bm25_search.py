import logging
from rank_bm25 import BM25Okapi

from .models import IndexedItem, SearchResult

logger = logging.getLogger(__name__)


class BM25Search:
    def __init__(self):
        self.index: BM25Okapi | None = None
        self.items: list[IndexedItem] = []
        self._tokenized: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def build(self, items: list[IndexedItem]) -> None:
        self.items = items
        self._tokenized = [self._tokenize(item.content) for item in items]
        self.index = BM25Okapi(self._tokenized)
        logger.info("BM25 index built with %d documents", len(items))

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if self.index is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:limit]
        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        out: list[SearchResult] = []
        for idx, score in ranked:
            item = self.items[idx]
            normalized = score / max_score
            out.append(
                SearchResult(
                    content=item.content,
                    source=item.source,
                    score=normalized,
                    source_type=item.source_type,
                    doc_meta=item.doc_meta,
                    project_meta=item.project_meta,
                )
            )
        return out
