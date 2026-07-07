import logging

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import Settings
from .models import DocMeta, IndexedItem, ProjectMeta, SearchResult, SourceType

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_url,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.settings.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", self.settings.collection_name)

    def index(self, items: list[IndexedItem]) -> None:
        texts = [item.content for item in items]
        embeddings = self.embeddings.embed_documents(texts)
        points = []
        for i, (item, emb) in enumerate(zip(items, embeddings, strict=True)):
            payload: dict = {
                "content": item.content,
                "source": item.source,
                "source_type": item.source_type.value,
            }
            if item.doc_meta:
                payload["doc_page"] = item.doc_meta.page
                payload["doc_section"] = item.doc_meta.section
                payload["doc_chunk_index"] = item.doc_meta.chunk_index
            if item.project_meta:
                payload["project_name"] = item.project_meta.name
            points.append(PointStruct(id=i, vector=emb, payload=payload))
        self.client.upsert(
            collection_name=self.settings.collection_name,
            points=points,
        )

    def search(self, query: str, limit: int | None = None) -> list[SearchResult]:
        if limit is None:
            limit = self.settings.top_k
        query_embedding = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.settings.collection_name,
            query_vector=query_embedding,
            limit=limit,
        )
        out: list[SearchResult] = []
        for r in results:
            p = r.payload or {}
            source_type = SourceType(p.get("source_type", "doc"))
            doc_meta = None
            project_meta = None
            if source_type == SourceType.DOC:
                doc_meta = DocMeta(
                    source=p.get("source", ""),
                    chunk_index=p.get("doc_chunk_index", 0),
                    page=p.get("doc_page"),
                    section=p.get("doc_section"),
                )
            else:
                project_meta = ProjectMeta(name=p.get("project_name", ""))
            out.append(
                SearchResult(
                    content=p.get("content", ""),
                    source=p.get("source", ""),
                    score=r.score,
                    source_type=source_type,
                    doc_meta=doc_meta,
                    project_meta=project_meta,
                )
            )
        return out

    def clear(self):
        self.client.delete_collection(self.settings.collection_name)
        self._ensure_collection()
