from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_mode: str = "console"  # console | vk | test | batch

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://localhost:11434"

    embed_model: str = "nomic-embed-text"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "alina_docs"

    postgres_url: str = "postgresql://alina:alina@localhost:5432/alina"

    vk_token: str = ""
    vk_group_id: str = ""

    docs_dir: str = "docs"
    projects_dir: str = "projects"
    tests_dir: str = "tests"

    chunk_size: int = 500
    chunk_overlap: int = 100

    chat_max_messages: int = 20
    chat_verbose: bool = False

    @property
    def ollama_host(self) -> str:
        return self.llm_base_url.rstrip("/")

    @property
    def docs_path(self) -> Path:
        return Path(self.docs_dir)

    @property
    def projects_path(self) -> Path:
        return Path(self.projects_dir)

    @property
    def tests_path(self) -> Path:
        return Path(self.tests_dir)


settings = Settings()
