import logging

import psycopg2
import psycopg2.extras

from alina_rag.models import ChunkRow

logger = logging.getLogger(__name__)


class Database:
    """Обёртка над PostgreSQL для хранения чанков и метаданных файлов."""

    def __init__(self, postgres_url: str):
        self._url = postgres_url

    def _conn(self):
        return psycopg2.connect(self._url)

    def init_tables(self) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (
                    source      TEXT PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    file_hash   TEXT NOT NULL,
                    indexed_at  TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          SERIAL PRIMARY KEY,
                    source      TEXT NOT NULL REFERENCES indexed_files(source) ON DELETE CASCADE,
                    filename    TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    chunk_text  TEXT NOT NULL,
                    UNIQUE(source, chunk_index)
                )
            """)
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm ON chunks USING gin (chunk_text gin_trgm_ops)")
            conn.commit()
        logger.info("Database tables initialized")

    def get_file_hashes(self) -> dict[str, str]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT source, file_hash FROM indexed_files")
            return dict(cur.fetchall())

    def upsert_file(self, source: str, filename: str, file_hash: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO indexed_files (source, filename, file_hash)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (source) DO UPDATE SET filename = EXCLUDED.filename, file_hash = EXCLUDED.file_hash, indexed_at = NOW()""",
                (source, filename, file_hash),
            )
            conn.commit()

    def delete_file(self, source: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM indexed_files WHERE source = %s", (source,))
            conn.commit()

    def insert_chunks(self, source: str, filename: str, texts: list[str]) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO chunks (source, filename, chunk_index, chunk_text) VALUES %s",
                [(source, filename, i, text) for i, text in enumerate(texts)],
            )
            conn.commit()

    def delete_chunks_by_source(self, source: str) -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
            conn.commit()

    def load_all_chunks(self) -> list[ChunkRow]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT id, source, filename, chunk_text, chunk_index FROM chunks ORDER BY source, chunk_index")
            return [ChunkRow(id=r[0], source=r[1], filename=r[2], chunk_text=r[3], chunk_index=r[4]) for r in cur.fetchall()]

    def trigram_search(self, query: str, top_k: int = 5) -> list[ChunkRow]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, source, filename, chunk_text, chunk_index
                   FROM chunks
                   ORDER BY similarity(chunk_text, %s) DESC
                   LIMIT %s""",
                (query, top_k),
            )
            return [ChunkRow(id=r[0], source=r[1], filename=r[2], chunk_text=r[3], chunk_index=r[4]) for r in cur.fetchall()]

    def get_chunk_ids_by_source(self, source: str) -> list[int]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM chunks WHERE source = %s ORDER BY chunk_index", (source,))
            return [row[0] for row in cur.fetchall()]
