import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

from alina_rag.models import ProjectRecord

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv"}

_LOADER_MAP = {
    ".txt": (TextLoader, {"autodetect_encoding": True}),
    ".md": (TextLoader, {"autodetect_encoding": True}),
    ".pdf": (PyMuPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
}


def load_document(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext not in _LOADER_MAP:
        logger.warning("Unsupported file type: %s", path)
        return []
    loader_cls, kwargs = _LOADER_MAP[ext]
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


def collect_files(*dirs: Path) -> list[Path]:
    files: list[Path] = []
    for root in dirs:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
    return files


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(path).encode())
    h.update(path.read_bytes())
    return h.hexdigest()


def load_projects(path: Path) -> list[ProjectRecord]:
    import pandas as pd
    records: list[ProjectRecord] = []
    project_files = [
        f for f in sorted(path.rglob("*"))
        if f.is_file() and f.suffix.lower() in {".xlsx", ".csv"}
    ]
    for fpath in project_files:
        try:
            if fpath.suffix.lower() == ".csv":
                df = pd.read_csv(fpath, encoding="utf-8")
            else:
                df = pd.read_excel(fpath)
        except Exception:
            logger.exception("Failed to read %s", fpath)
            continue
        name_col = None
        desc_col = None
        for col in df.columns:
            cl = str(col).strip().lower()
            if cl in ("name", "название", "имя", "проект"):
                name_col = col
            elif cl in ("description", "описание", "descr"):
                desc_col = col
        if name_col is None and desc_col is None:
            logger.warning("No name/description columns in %s, skipping", fpath.name)
            continue
        for _, row in df.iterrows():
            name = str(row[name_col]).strip() if name_col else ""
            desc = str(row[desc_col]).strip() if desc_col else ""
            if not name or name == "nan":
                continue
            if desc == "nan":
                desc = ""
            records.append(ProjectRecord(name=name, description=desc, file=fpath.name))
    return records
