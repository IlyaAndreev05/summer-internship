"""ChromaDB-based vector store implementation."""

import asyncio

import chromadb
from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction

from alina_rag.config import Settings
from alina_rag.domain.interfaces import EmbeddingProvider, VectorStore
from alina_rag.domain.models import SearchResult


class _EmbeddingAdapter(ChromaEmbeddingFunction):
    """Adapts async EmbeddingProvider to ChromaDB's sync EmbeddingFunction."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        """Synchronous wrapper that runs the async embed in a new event loop."""
        return asyncio.run(self._provider.embed(input))


class ChromaStore(VectorStore):
    """ChromaDB-backed vector store for semantic search over document chunks."""

    def __init__(self, settings: Settings, embed_provider: EmbeddingProvider) -> None:
        self._settings = settings
        self._embed_provider = embed_provider
        self._client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=_EmbeddingAdapter(embed_provider),
        )

    async def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        metadata: list[dict[str, str]],
    ) -> None:
        """Embed and store document chunks."""
        if not chunks:
            return
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        embeddings = await self._embed_provider.embed(chunks)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
        )

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for chunks semantically similar to the query."""
        query_embedding = await self._embed_provider.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        search_results: list[SearchResult] = []
        ids_list = results.get("ids", [[]])[0]
        documents_list = results.get("documents", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids_list):
            doc_meta = metadatas_list[i] if i < len(metadatas_list) else {}
            doc_id_from_meta = doc_meta.get("document_id", "")
            score = 1.0 - distances_list[i] if i < len(distances_list) else 0.0
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=doc_id_from_meta,
                    content=documents_list[i] if i < len(documents_list) else "",
                    metadata=doc_meta,
                    score=score,
                )
            )
        return search_results

    async def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        existing = self._collection.get(
            where={"document_id": doc_id},
        )
        chunk_ids = existing.get("ids", [])
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

    async def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()
