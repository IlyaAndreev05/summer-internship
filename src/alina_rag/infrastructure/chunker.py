"""Text chunking strategies — OOP hierarchy with pluggable implementations."""

from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract chunker: splits text into overlapping chunks.

    Subclass and override chunk() to implement a new strategy.
    """

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks. Each chunk is a string."""
        ...


class ParagraphChunker(BaseChunker):
    """Chunk by paragraph boundaries with configurable size and overlap.

    Paragraphs (split by double-newline) are grouped until they exceed
    ``chunk_size`` characters. Each chunk overlaps the previous one by
    ``overlap`` characters from the tail of the preceding chunk.

    This is the default chunker — fast, no external dependencies.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self._chunk_size and current:
                chunk = "\n\n".join(current)
                chunks.append(chunk)
                # Build overlap from the tail of the finished chunk
                overlap_text = (
                    chunk[-self._overlap :]
                    if len(chunk) > self._overlap
                    else chunk
                )
                current = [overlap_text]
                current_len = len(overlap_text)
            current.append(para)
            current_len += para_len + 2  # +2 for "\n\n"

        if current:
            chunks.append("\n\n".join(current))

        return chunks


class SemanticChunker(BaseChunker):
    """Chunk by semantic boundaries using embedding similarity.

    Splits text into sentences, then groups consecutive sentences
    while their cosine similarity stays above a threshold.
    When similarity drops → new chunk boundary.

    Requires an EmbeddingProvider — heavier but produces coherent chunks.
    """

    def __init__(
        self,
        embed_provider,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
    ) -> None:
        self._embed = embed_provider
        self._threshold = threshold
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks."""
        import re
        import numpy as np

        # Split into sentences (Russian-aware: split on .!? followed by space/end)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Get embeddings (sync call — ok for small batches)
        embeddings = self._embed.encode(sentences)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            sent_len = len(sentences[i])

            # New chunk if similarity drops AND we're above min size
            if sim < self._threshold and current_len >= self._min_chunk_size:
                chunks.append(" ".join(current))
                current = [sentences[i]]
                current_len = sent_len
            # Also split if exceeding max size
            elif current_len + sent_len > self._max_chunk_size and current_len >= self._min_chunk_size:
                chunks.append(" ".join(current))
                current = [sentences[i]]
                current_len = sent_len
            else:
                current.append(sentences[i])
                current_len += sent_len + 1  # +1 for space

        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def _cosine_sim(a, b) -> float:
        """Cosine similarity between two vectors."""
        import numpy as np
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
