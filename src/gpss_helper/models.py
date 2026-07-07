from dataclasses import dataclass
from enum import Enum


class SourceType(Enum):
    DOC = "doc"
    PROJECT = "project"


@dataclass
class DocMeta:
    source: str
    chunk_index: int
    page: int | None = None
    section: str | None = None


@dataclass
class ProjectMeta:
    name: str


@dataclass
class SearchResult:
    content: str
    source: str
    score: float
    source_type: SourceType
    doc_meta: DocMeta | None = None
    project_meta: ProjectMeta | None = None


@dataclass
class IndexedItem:
    content: str
    source: str
    source_type: SourceType
    doc_meta: DocMeta | None = None
    project_meta: ProjectMeta | None = None
