from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:1.5b"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    collection_name: str = "gpss_rag"
    docs_dir: str = "data/docs"
    projects_dir: str = "data/projects"
    batch_input_dir: str = "data/batch/input"
    batch_output_dir: str = "data/batch/output"
    test_input_dir: str = "data/test/input"
    test_output_dir: str = "data/test/output"
    vk_token: str = ""
    vk_group_id: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
