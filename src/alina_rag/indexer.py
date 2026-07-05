import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

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

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {
    ".txt": (TextLoader, {"autodetect_encoding": True}),
    ".md": (TextLoader, {"autodetect_encoding": True}),
    ".pdf": (PyMuPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
}


def _load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        logger.warning("Unsupported file type: %s", path)
        return []
    loader_cls, kwargs = SUPPORTED_EXTS[ext]
    try:
        loader = loader_cls(str(path), **kwargs)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(path)
            doc.metadata["filename"] = path.name
        return docs
    except Exception:
        logger.exception("Failed to load %s", path)
        return []


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(path).encode())
    h.update(path.read_bytes())
    return h.hexdigest()


def _collect_files() -> list[Path]:
    files: list[Path] = []
    for root in [settings.docs_path, settings.projects_path]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
    return files


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_host)


def _ensure_collection(client: QdrantClient):
    try:
        client.get_collection(settings.qdrant_collection)
    except Exception:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )


def _sync_vectors() -> None:
    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    _ensure_collection(client)

    client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    rows = load_all_chunks()
    if not rows:
        return

    texts = [r[3] for r in rows]
    vectors = embeddings.embed_documents(texts)
    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "source": r[1],
                "filename": r[2],
                "chunk_index": r[4],
            },
        )
        for i, r in enumerate(rows)
    ]

    batch_size = 100
    total = len(points)
    print(f"  ⏳ Синхронизация векторов: 0/{total}", end="", flush=True)
    for i in range(0, total, batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)
        done = min(i + batch_size, total)
        print(f"\r  ⏳ Синхронизация векторов: {done}/{total}", end="", flush=True)
        logger.info("Synced vectors: %d/%d", done, total)
    print()

    logger.info("Vector sync complete: %d points", total)


def auto_index() -> None:
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
            fh = _file_hash(path)
        except Exception:
            logger.exception("Failed to hash %s", path)
            continue
        current[source] = fh
        if source not in registry:
            new_files.append(path)
        elif registry[source] != fh:
            changed_files.append(path)

    deleted_sources = set(registry) - set(current)

    if not new_files and not changed_files and not deleted_sources:
        pg_count = len(load_all_chunks())
        client = _get_qdrant_client()
        try:
            qdrant_count = client.get_collection(settings.qdrant_collection).points_count
        except Exception:
            qdrant_count = 0
        if pg_count > 0 and qdrant_count < pg_count:
            logger.warning("Qdrant mismatch (%d pg vs %d qdrant), re-syncing", pg_count, qdrant_count)
            _sync_vectors()
            return
        logger.info("All %d files up to date", len(files))
        return

    logger.info("Indexing: %d new, %d changed, %d deleted", len(new_files), len(changed_files), len(deleted_sources))

    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    _ensure_collection(client)

    for source in list(deleted_sources) + [str(p) for p in changed_files]:
        try:
            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
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
            print(f"  ⏭  {path.name}: не удалось загрузить")
            continue

        chunks = splitter.split_documents(docs)
        texts = [doc.page_content for doc in chunks]
        n = len(texts)
        print(f"  📄 {path.name}: {n} чанков")

        upsert_file(source, path.name, _file_hash(path))
        delete_chunks_by_source(source)
        insert_chunks(source, path.name, texts)

        vectors = embeddings.embed_documents(texts)
        start_id = total_chunks
        points = [
            PointStruct(
                id=start_id + i,
                vector=vectors[i],
                payload={
                    "source": source,
                    "filename": path.name,
                    "chunk_index": i,
                },
            )
            for i in range(n)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total_chunks += n
        print(f"  ✅ всего чанков: {total_chunks}")

    print(f"\nГотово: {total_chunks} чанков из {len(to_index)} файлов")
    logger.info("Indexed %d chunks from %d files", total_chunks, len(to_index))
