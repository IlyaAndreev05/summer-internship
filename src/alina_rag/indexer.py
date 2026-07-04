import logging
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {
    ".txt": (TextLoader, {"autodetect_encoding": True}),
    ".md": (TextLoader, {"autodetect_encoding": True}),
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
}


def _load_file(path: Path) -> list[Document]:
    """Загружает файл по расширению и возвращает список документов."""
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
