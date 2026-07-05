from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_mode: str = "console"

    llm_model: str = "qwen2.5:1.5b"
    llm_base_url: str = "http://localhost:11434"

    embed_model: str = "nomic-embed-text"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "alina_docs"

    postgres_url: str = "postgresql://alina:alina@localhost:5432/alina"

    vk_token: str = ""
    vk_group_id: str = ""

    data_dir: str = "data"

    chunk_size: int = 500
    chunk_overlap: int = 100

    chat_verbose: bool = False

    @property
    def ollama_host(self) -> str:
        return self.llm_base_url.rstrip("/")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def docs_path(self) -> Path:
        return Path("data/docs")

    @property
    def projects_path(self) -> Path:
        return Path("data/projects")

    @property
    def questions_input_path(self) -> Path:
        return Path("data/questions/input")

    @property
    def questions_output_path(self) -> Path:
        return Path("data/questions/output")


settings = Settings()
