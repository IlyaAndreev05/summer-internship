"""Application configuration via env vars / .env file."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────
    llm_provider: Literal["ollama", "openai", "deepseek"] = "ollama"
    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""

    # ── Embeddings ────────────────────────────────────
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embed_device: str = "cpu"

    # ── SQLite ────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///data/chat_history.db"

    # ── ChromaDB ──────────────────────────────────────
    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "alina_docs"

    # ── Telegram ──────────────────────────────────────
    telegram_token: str = ""

    # ── VK ────────────────────────────────────────────
    vk_token: str = ""
    vk_group_id: str = ""

    # ── Chat ──────────────────────────────────────────
    chat_max_messages: int = 20

    # ── Prompts ────────────────────────────────────────
    # Override system prompt: set a string directly, or @path/to/file.txt
    agent_system_prompt: str = ""
    # Extra context appended after system prompt (templates, greetings, rules)
    agent_extra_context: str = ""

    # ── Chunking ───────────────────────────────────────
    chunker_type: Literal["paragraph", "semantic"] = "paragraph"
    chunker_size: int = 500
    chunker_overlap: int = 100

    # ── Documents ─────────────────────────────────────
    docs_dir: str = "data/documents"

    @property
    def ollama_host(self) -> str:
        """Extract Ollama host from base URL for the Ollama client."""
        # http://localhost:11434/v1 → http://localhost:11434
        return self.llm_base_url.rstrip("/v1").rstrip("/")

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def docs_path(self) -> Path:
        return Path(self.docs_dir)


# Singleton
settings = Settings()
