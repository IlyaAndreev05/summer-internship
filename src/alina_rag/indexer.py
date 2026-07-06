from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

from alina_rag.db import Database
from alina_rag.loaders import collect_files, file_hash, load_document, load_projects
from alina_rag.models import IndexProgress, IndexStats

logger = logging.getLogger(__name__)

_PROJECT_BATCH = 200
_EMBED_BATCH = 50


class Indexer:
    """Полный конвейер индексации: файлы → чанки → эмбеддинги → Qdrant."""

    def __init__(
        self,
        db: Database,
        qdrant_url: str,
        collection: str,
        embed_model: str,
        ollama_host: str,
    ):
        self._db = db
        self._client = QdrantClient(url=qdrant_url)
        self._collection = collection
        self._embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_host)
        self._ready = True
        self._stats: IndexStats | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def stats(self) -> IndexStats | None:
        return self._stats

    def index(
        self,
        docs_path: Path,
        projects_path: Path,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> IndexStats:
        self._ready = False
        start = time.time()

        try:
            self._db.init_tables()

            files = collect_files(docs_path, projects_path)
            registry = self._db.get_file_hashes()
            current: dict[str, str] = {}

            new_files: list[Path] = []
            changed_files: list[Path] = []

            for path in files:
                source = str(path)
                try:
                    fh = file_hash(path)
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
                pg_count = len(self._db.load_all_chunks())
                try:
                    qdrant_count = self._client.count(
                        collection_name=self._collection, exact=True
                    ).count
                except Exception:
                    qdrant_count = 0
                if pg_count > 0 and qdrant_count < pg_count:
                    logger.warning(
                        "Qdrant mismatch (%d pg vs %d qdrant), re-syncing",
                        pg_count,
                        qdrant_count,
                    )
                    self._sync_vectors()
                end = time.time()
                self._stats = IndexStats(
                    total_files=len(files),
                    total_chunks=pg_count,
                    new_files=0,
                    changed_files=0,
                    deleted_files=0,
                    start_time=start,
                    end_time=end,
                )
                print(f"\n✅ Все {len(files)} файлов актуальны. "
                      f"Чанков в базе: {pg_count}. "
                      f"Проверено за {self._stats.elapsed:.1f}с")
                return self._stats

            logger.info(
                "Indexing: %d new, %d changed, %d deleted",
                len(new_files),
                len(changed_files),
                len(deleted_sources),
            )

            self._ensure_collection()

            for source in list(deleted_sources) + [str(p) for p in changed_files]:
                try:
                    self._client.delete(
                        collection_name=self._collection,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="source", match=MatchValue(value=source)
                                )
                            ]
                        ),
                    )
                except Exception:
                    logger.exception("Failed to remove vectors for: %s", source)

            for source in deleted_sources:
                self._db.delete_file(source)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            total_chunks = 0
            to_index = new_files + changed_files
            files_done = 0
            est_total_chunks = max(len(to_index) * 10, 1)

            for path in to_index:
                source = str(path)
                docs = load_document(path)
                if not docs:
                    print(f"  ⏭  {path.name}: не удалось загрузить")
                    continue

                chunks = splitter.split_documents(docs)
                texts = [doc.page_content for doc in chunks]
                n = len(texts)
                files_done += 1

                self._db.upsert_file(source, path.name, file_hash(path))
                self._db.delete_chunks_by_source(source)
                self._db.insert_chunks(source, path.name, texts)

                vectors = self._embed_parallel(texts)
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
                self._client.upsert(
                    collection_name=self._collection, points=points
                )
                total_chunks += n

                if files_done > 0:
                    est_total_chunks = int(total_chunks / files_done * len(to_index))

                progress = IndexProgress(
                    current=files_done,
                    total=len(to_index),
                    file_name=path.name,
                    chunks_done=total_chunks,
                    start_time=start,
                )
                print(f"\r  📄 {files_done}/{len(to_index)} файлов | "
                      f"{total_chunks} чанков | "
                      f"{progress.speed:.1f} чанков/сек | "
                      f"прошло: {progress.elapsed:.0f}с", end="", flush=True)

            if projects_path.exists():
                project_records = load_projects(projects_path)
                if project_records:
                    texts = []
                    sources = []
                    for i, rec in enumerate(project_records):
                        text = f"Проект: {rec.name}\nОписание: {rec.description}"
                        texts.append(text)
                        sources.append(f"project:{rec.file}:{i}")

                    total = len(texts)
                    print(
                        f"\r  ⏳ Индексация проектов: 0/{total}",
                        end="",
                        flush=True,
                    )

                    done = 0
                    for batch_start in range(0, total, _PROJECT_BATCH):
                        batch_end = min(batch_start + _PROJECT_BATCH, total)
                        batch_texts = texts[batch_start:batch_end]
                        batch_sources = sources[batch_start:batch_end]

                        for j, src in enumerate(batch_sources):
                            parts = src.split(":")
                            filename = parts[1] if len(parts) > 1 else "unknown"
                            self._db.upsert_file(src, filename, "project")
                            self._db.delete_chunks_by_source(src)
                            self._db.insert_chunks(src, filename, [batch_texts[j]])

                        vectors = self._embed_parallel(batch_texts)
                        start_id = total_chunks
                        points = [
                            PointStruct(
                                id=start_id + j,
                                vector=vectors[j],
                                payload={
                                    "source": batch_sources[j],
                                    "filename": "project",
                                    "chunk_index": 0,
                                },
                            )
                            for j in range(len(batch_texts))
                        ]
                        self._client.upsert(
                            collection_name=self._collection, points=points
                        )
                        total_chunks += len(batch_texts)
                        done += len(batch_texts)
                        print(
                            f"\r  ⏳ Индексация проектов: {done}/{total}",
                            end="",
                            flush=True,
                        )

                    print()

            pg_count = len(self._db.load_all_chunks())
            try:
                qdrant_count = self._client.count(
                    collection_name=self._collection, exact=True
                ).count
            except Exception:
                qdrant_count = 0

            if pg_count > 0 and qdrant_count < pg_count:
                logger.warning(
                    "Qdrant mismatch after indexing (%d pg vs %d qdrant), re-syncing",
                    pg_count,
                    qdrant_count,
                )
                self._sync_vectors()

            end = time.time()
            self._stats = IndexStats(
                total_files=len(to_index),
                total_chunks=total_chunks,
                new_files=len(new_files),
                changed_files=len(changed_files),
                deleted_files=len(deleted_sources),
                start_time=start,
                end_time=end,
            )
            print(f"\n✅ Индексация завершена:\n   Файлов: {self._stats.total_files} "
                  f"(новых: {self._stats.new_files}, изменено: {self._stats.changed_files}, "
                  f"удалено: {self._stats.deleted_files})\n   Чанков: {self._stats.total_chunks}\n"
                  f"   Время: {self._stats.elapsed:.1f}с\n   Скорость: {self._stats.avg_speed:.1f} чанков/сек")
            logger.info("Indexed %d chunks from %d files", total_chunks, len(to_index))
            return self._stats
        finally:
            self._ready = True

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=768, distance=Distance.COSINE
                ),
            )

    def _sync_vectors(self) -> None:
        rows = self._db.load_all_chunks()
        if not rows:
            return

        total = len(rows)
        print(f"  ⏳ Синхронизация векторов: 0/{total}", end="", flush=True)

        batches = [
            (i, rows[i : i + _EMBED_BATCH])
            for i in range(0, total, _EMBED_BATCH)
        ]
        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self._embed_chunk_batch, s, b): s
                for s, b in batches
            }
            for future in as_completed(futures):
                _, points = future.result()
                self._client.upsert(
                    collection_name=self._collection, points=points
                )
                done += len(points)
                print(
                    f"\r  ⏳ Синхронизация векторов: {done}/{total}",
                    end="",
                    flush=True,
                )
                logger.info("Synced vectors: %d/%d", done, total)
        print()

    def _embed_chunk_batch(
        self, start: int, batch_rows: list
    ) -> tuple[int, list[PointStruct]]:
        texts = [r.chunk_text for r in batch_rows]
        vectors = self._embeddings.embed_documents(texts)
        points = [
            PointStruct(
                id=start + j,
                vector=vectors[j],
                payload={
                    "source": r.source,
                    "filename": r.filename,
                    "chunk_index": r.chunk_index,
                },
            )
            for j, r in enumerate(batch_rows)
        ]
        return start, points

    def _embed_parallel(self, texts: list[str]) -> list:
        if not texts:
            return []
        batches = [
            (i, texts[i : i + _EMBED_BATCH])
            for i in range(0, len(texts), _EMBED_BATCH)
        ]
        results: list[tuple[int, list]] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self._embed_text_batch, s, b): s
                for s, b in batches
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda x: x[0])
        all_vectors: list = []
        for _, vectors in results:
            all_vectors.extend(vectors)
        return all_vectors

    def _embed_text_batch(
        self, start: int, texts: list[str]
    ) -> tuple[int, list]:
        vectors = self._embeddings.embed_documents(texts)
        return start, vectors
