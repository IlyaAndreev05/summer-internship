"""Document ingestion orchestration — parse, chunk, embed, store in dual index."""

from pathlib import Path
from typing import TYPE_CHECKING

from alina_rag.domain.models import DocSource, Document
from alina_rag.infrastructure.document_parser import parse_file
from alina_rag.infrastructure.chunker import BaseChunker, ParagraphChunker

if TYPE_CHECKING:
    from alina_rag.domain.interfaces import EmbeddingProvider, VectorStore
    from alina_rag.infrastructure.bm25_store import BM25Store


class DocumentService:
    """Ingests documents into dual index: vector store + BM25 keyword store.

    Chunking strategy is pluggable via the ``chunker`` parameter.
    Default: ParagraphChunker (500 chars, 100 overlap).
    Switch to SemanticChunker for better quality on structured documents.
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        embed_provider: "EmbeddingProvider",
        bm25_store: "BM25Store | None" = None,
        chunker: BaseChunker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embed_provider = embed_provider
        self._bm25_store = bm25_store
        self._chunker = chunker or ParagraphChunker()

    async def ingest_file(self, path: Path) -> int:
        """Parse a file, chunk it, store in vector + BM25. Returns chunk count."""
        text = parse_file(path)
        if not text.strip():
            return 0

        chunks = self._chunker.chunk(text)
        if not chunks:
            return 0

        doc = Document(
            source=DocSource.MANUAL,
            filename=path.name,
            title=path.stem,
            content=text,
            chunk_count=len(chunks),
        )

        metadata = [{"filename": path.name, "title": path.stem} for _ in chunks]

        # Vector index (embeddings)
        await self._vector_store.add_chunks(doc.id, chunks, metadata)

        # BM25 keyword index
        if self._bm25_store is not None:
            self._bm25_store.add_chunks(doc.id, chunks, metadata)

        return len(chunks)

    async def ingest_directory(self, dir_path: Path) -> int:
        """Ingest all supported files from a directory. Returns total chunk count."""
        total = 0
        supported_exts = {".txt", ".pdf", ".docx", ".md", ".xlsx", ".xls", ".csv", ".log"}

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                total += await self.ingest_file(file_path)

        return total

    async def list_documents(self) -> list[Document]:
        """Return list of ingested documents from the vector store."""
        return []
