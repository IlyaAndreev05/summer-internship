import logging
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from alina_rag.agent import RAGAgent
from alina_rag.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {
    ".txt": (TextLoader, {"autodetect_encoding": True}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyPDFLoader, {}),
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


def _walk_and_load(root: Path) -> list[Document]:
    docs: list[Document] = []
    if not root.exists():
        logger.warning("Directory not found: %s", root)
        return docs
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            docs.extend(_load_file(path))
    return docs


def index_documents(agent: RAGAgent) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs: list[Document] = []
    all_docs.extend(_walk_and_load(settings.docs_path))
    all_docs.extend(_walk_and_load(settings.projects_path))

    if not all_docs:
        logger.warning("No documents found to index")
        return 0

    chunks = splitter.split_documents(all_docs)

    vector_store = agent.get_vector_store()
    bm25_store = agent.get_bm25_store()

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    vector_store.add_documents(chunks)

    if texts:
        bm25_store.add_chunks(texts, metadatas)

    logger.info("Indexed %d chunks from %d documents", len(chunks), len(all_docs))
    return len(chunks)
