import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from alina_rag.agent import RAGAgent
from alina_rag.config import settings
from alina_rag.db import (
    delete_chunks_by_source,
    delete_file,
    get_file_hashes,
    init_tables,
    insert_chunks,
    load_all_chunks,
    upsert_file,
)
from alina_rag.indexer import SUPPORTED_EXTS, _load_file

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    """Вычисляет sha256 хеш пути + содержимого файла."""
    h = hashlib.sha256()
    h.update(str(path).encode())
    h.update(path.read_bytes())
    return h.hexdigest()


def _collect_files() -> list[Path]:
    """Обходит data/ и projects/, возвращает список поддерживаемых файлов."""
    files: list[Path] = []
    for root in [settings.data_path, settings.projects_path]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
    return files


def _sync_vectors(expected_count: int) -> None:
    """Перезапись чанков из Postgres в Qdrant."""
    agent = RAGAgent()
    vector_store = agent.get_vector_store()

    vector_store.client.delete_collection(settings.qdrant_collection)
    vector_store.client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    rows = load_all_chunks()
    docs = [
        Document(page_content=r[3], metadata={"source": r[1], "filename": r[2]})
        for r in rows
    ]

    batch_size = 100
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(batch)
        done = min(i + batch_size, total)
        remaining = total - done
        logger.info("Synced vectors: %d/%d (осталось %d)", done, total, remaining)
        print(f"\r  ⏳ Векторы: {done}/{total} (осталось {total - done})", end="", flush=True)

    print()
    logger.info("Vector sync complete: %d points inserted into Qdrant", total)

def auto_index() -> None:
    """Индексирует новые/изменённые файлы с дедупликацией через Postgres."""
    init_tables()

    files = _collect_files()
    if not files:
        logger.info("No documents found to index")
        return

    registry = get_file_hashes()
    current: dict[str, str] = {}

    new_files: list[Path] = []
    changed_files: list[Path] = []

    for path in files:
        source = str(path)
        try:
            file_hash = _file_hash(path)
        except Exception:
            logger.exception("Failed to hash %s", path)
            continue

        current[source] = file_hash

        if source not in registry:
            new_files.append(path)

    deleted_sources = set(registry) - set(current)

    if not new_files and not changed_files and not deleted_sources:
        pg_count = len(load_all_chunks())
        client = QdrantClient(url=settings.qdrant_url)
        try:
            qdrant_count = client.get_collection(settings.qdrant_collection).points_count
        except Exception:
            qdrant_count = 0

        if pg_count > 0 and qdrant_count < pg_count:
            logger.warning(
                "Qdrant vector mismatch: %d chunks in Postgres vs %d points in Qdrant. Re-syncing vectors.",
                pg_count, qdrant_count,
            )
            _sync_vectors(pg_count)
            return

        logger.info("All %d files up to date", len(files))
        return

    logger.info("Indexing: %d new, %d changed, %d deleted", len(new_files), len(changed_files), len(deleted_sources))


    agent = RAGAgent()
    vector_store = agent.get_vector_store()

    for source in list(deleted_sources) + [str(p) for p in changed_files]:
        try:
            vector_store.client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=Filter(
                    must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
                ),
            )
        except Exception:
            logger.exception("Failed to remove vectors for: %s", source)

    for source in deleted_sources:
        delete_file(source)

    to_index = new_files + changed_files
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0
    for path in to_index:
        source = str(path)
        docs = _load_file(path)
        if not docs:
            continue

        chunks = splitter.split_documents(docs)
        texts = [doc.page_content for doc in chunks]

        upsert_file(source, path.name, _file_hash(path))
        delete_chunks_by_source(source)
        insert_chunks(source, path.name, texts)

        vector_store.add_documents(chunks)

        total_chunks += len(chunks)

    logger.info("Indexed %d chunks from %d files", total_chunks, len(to_index))
