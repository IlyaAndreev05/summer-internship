from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Результат поиска до разрешения текста."""

    id: str
    score: float
    source: str
    chunk_index: int


@dataclass(frozen=True, slots=True)
class ScoredResult:
    """Результат поиска с разрешённым текстом."""

    id: str
    score: float
    source: str
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    """Одна строка из таблицы проектов."""

    name: str
    description: str
    file: str


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """Строка чанка из базы данных."""

    id: int
    source: str
    filename: str
    chunk_text: str
    chunk_index: int


@dataclass
class ChatMessage:
    """Единое сообщение в истории диалога."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatHistory:
    """Менеджер истории диалога."""

    def __init__(self, max_turns: int = 20) -> None:
        self._messages: list[ChatMessage] = []
        self._max_turns = max_turns

    def add_user(self, content: str) -> None:
        self._messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(ChatMessage(role="assistant", content=content))

    def last_dicts(self, n: int | None = None) -> list[dict[str, str]]:
        count = n or self._max_turns
        return [m.to_dict() for m in self._messages[-count:]]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


@dataclass
class IndexStats:
    """Результат индексации."""

    total_files: int
    total_chunks: int
    new_files: int
    changed_files: int
    deleted_files: int
    start_time: float
    end_time: float

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time

    @property
    def avg_speed(self) -> float:
        return self.total_chunks / self.elapsed if self.elapsed > 0 else 0.0


@dataclass
class IndexProgress:
    """Снимок прогресса во время индексации."""

    current: int
    total: int
    file_name: str
    chunks_done: int
    start_time: float

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def speed(self) -> float:
        return self.chunks_done / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def eta_seconds(self) -> float:
        remaining = self.total - self.chunks_done
        spd = self.speed
        return remaining / spd if spd > 0 else 0.0

    def format_progress(self) -> str:
        elapsed = self.elapsed
        spd = self.speed
        eta = self.eta_seconds
        return (
            f"⏳ {self.chunks_done}/{self.total} чанков "
            f"| {spd:.1f} чанков/сек "
            f"| прошло: {elapsed:.0f}с "
            f"| ETA: {eta:.0f}с"
        )

    def format_done(self, stats: IndexStats) -> str:
        return (
            f"\n✅ Индексация завершена:\n"
            f"   Файлов: {stats.total_files} (новых: {stats.new_files}, "
            f"изменено: {stats.changed_files}, удалено: {stats.deleted_files})\n"
            f"   Чанков: {stats.total_chunks}\n"
            f"   Время: {stats.elapsed:.1f}с\n"
            f"   Скорость: {stats.avg_speed:.1f} чанков/сек"
        )
