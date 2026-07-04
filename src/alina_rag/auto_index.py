import hashlib
import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from alina_rag.config import settings
from alina_rag.indexer import SUPPORTED_EXTS, _load_file

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("data/index_registry.json")


def _file_hash(path: Path) -> str:
    """Compute sha256 of file path + content."""
    h = hashlib.sha256()
    h.update(str(path).encode())
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_registry() -> dict[str, str]:
    """Load {source_path: hash} from registry file."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except Exception:
        logger.warning("Failed to read registry, starting fresh")
        return {}


def _save_registry(registry: dict[str, str]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2))


def _collect_files() -> list[Path]:
    """Walk docs/ and projects/, return sorted list of supported files."""
    files: list[Path] = []
    for root in [settings.docs_path, settings.projects_path]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
    return files


def auto_index() -> None:
    """Auto-index new/changed files. Skips already-indexed files by hash+path."""
    from alina_rag.agent import RAGAgent

    files = _collect_files()
    if not files:
        logger.info("No documents found to index")
        return

    registry = _load_registry()
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
        elif registry[source] != file_hash:
            changed_files.append(path)
        # else: unchanged, skip

    # Detect deleted files
    deleted_sources = set(registry) - set(current)

    if not new_files and not changed_files and not deleted_sources:
        logger.info("All %d files up to date, nothing to index", len(files))
        return

    logger.info(
        "Indexing: %d new, %d changed, %d deleted",
        len(new_files), len(changed_files), len(deleted_sources),
    )

    agent = RAGAgent()

    # Remove chunks for changed and deleted files
    for source in deleted_sources:
        try:
            agent.remove_by_source(source)
        except Exception:
            logger.exception("Failed to remove chunks for deleted file: %s", source)

    for path in changed_files:
        try:
            agent.remove_by_source(str(path))
        except Exception:
            logger.exception("Failed to remove old chunks for: %s", path)

    # Index new and changed files
    to_index = new_files + changed_files
    if to_index:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_docs: list[Document] = []
        for path in to_index:
            all_docs.extend(_load_file(path))

        if all_docs:
            chunks = splitter.split_documents(all_docs)
            vector_store = agent.get_vector_store()
            bm25_store = agent.get_bm25_store()

            vector_store.add_documents(chunks)

            texts = [doc.page_content for doc in chunks]
            metadatas = [doc.metadata for doc in chunks]
            if texts:
                bm25_store.add_chunks(texts, metadatas)

            logger.info("Indexed %d chunks from %d files", len(chunks), len(to_index))

    # Update registry (remove deleted, add new/changed)
    for source in deleted_sources:
        registry.pop(source, None)
    registry.update(current)

    _save_registry(registry)
    logger.info("Registry updated: %d files tracked", len(registry))
