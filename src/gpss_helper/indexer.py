import time
import logging
import threading
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .models import IndexedItem, SourceType, DocMeta, ProjectMeta
from .vector_store import QdrantVectorStore
from .bm25_search import BM25Search

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self, settings: Settings, vector_store: QdrantVectorStore, bm25: BM25Search
    ):
        self.settings = settings
        self.vector_store = vector_store
        self.bm25 = bm25
        self._indexed = False
        self._error: Exception | None = None
        self._thread: threading.Thread | None = None

    @property
    def indexed(self) -> bool:
        return self._indexed

    @property
    def error(self) -> Exception | None:
        return self._error

    def _read_pdf(self, path: Path) -> str:
        parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def _read_docx(self, path: Path) -> str:
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text)

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def _load_document(self, path: Path) -> list[IndexedItem]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(path)
        elif suffix == ".docx":
            text = self._read_docx(path)
        elif suffix in (".txt", ".md"):
            text = self._read_text(path)
        else:
            logger.warning("Skipping unsupported file: %s", path)
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        items: list[IndexedItem] = []
        for i, chunk in enumerate(chunks):
            items.append(
                IndexedItem(
                    content=chunk,
                    source=str(path),
                    source_type=SourceType.DOC,
                    doc_meta=DocMeta(source=str(path), chunk_index=i),
                )
            )
        return items

    def _load_projects_file(self, path: Path) -> list[IndexedItem]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".xlsx":
            df = pd.read_excel(path)
        elif suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            logger.warning("Skipping unsupported project file: %s", path)
            return []

        items: list[IndexedItem] = []
        for _, row in df.iterrows():
            name = str(
                row.get("name", row.get("Name", row.get("Название", "")))
            )
            description = str(
                row.get(
                    "description",
                    row.get("Description", row.get("Описание", "")),
                )
            )
            content = f"{name}: {description}"
            items.append(
                IndexedItem(
                    content=content,
                    source=str(path),
                    source_type=SourceType.PROJECT,
                    project_meta=ProjectMeta(name=name),
                )
            )
        return items

    def _run_indexing(self) -> None:
        start = time.time()
        docs_dir = Path(self.settings.docs_dir)
        projects_dir = Path(self.settings.projects_dir)

        doc_files: list[Path] = []
        for ext in ("*.pdf", "*.txt", "*.md", "*.docx"):
            doc_files.extend(docs_dir.glob(ext))

        project_files: list[Path] = []
        for ext in ("*.csv", "*.xlsx", "*.jsonl"):
            project_files.extend(projects_dir.glob(ext))

        all_items: list[IndexedItem] = []

        for f in doc_files:
            items = self._load_document(f)
            all_items.extend(items)
            logger.info(
                "Loaded %d chunks from %s (processed %d total)",
                len(items),
                f.name,
                len(all_items),
            )

        for f in project_files:
            items = self._load_projects_file(f)
            all_items.extend(items)
            logger.info(
                "Loaded %d projects from %s (processed %d total)",
                len(items),
                f.name,
                len(all_items),
            )

        total = len(all_items)
        if total == 0:
            logger.warning("Nothing to index")
            self._indexed = True
            return

        logger.info("Indexing %d items into Qdrant...", total)
        t0 = time.time()
        self.vector_store.index(all_items)
        t1 = time.time()
        logger.info("Qdrant indexing done in %.1fs", t1 - t0)

        logger.info("Building BM25 index...")
        self.bm25.build(all_items)

        elapsed = time.time() - start
        speed = total / elapsed if elapsed > 0 else 0
        logger.info(
            "Indexed %d items in %.1fs (%.0f items/s)",
            total,
            elapsed,
            speed,
        )

        if elapsed > 120:
            logger.warning(
                "Indexing took %.1fs, exceeding 2 minute target", elapsed
            )

        self._indexed = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._safe_run, daemon=False)
        self._thread.start()

    def _safe_run(self) -> None:
        try:
            self._run_indexing()
        except Exception as e:
            self._error = e
            logger.critical("Indexing failed: %s", e, exc_info=True)
            raise SystemExit(1) from e
