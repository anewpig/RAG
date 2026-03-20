from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""

    embedding_provider: str = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    chat_model: str = "gpt-4.1-mini"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "policylens_chunks"

    data_dir: str = "./data"
    app_env: str = "dev"

    chunk_size: int = 400
    chunk_overlap: int = 60
    embedding_batch_size: int = 64

    retrieval_top_k: int = 5
    generation_max_chunks: int = 4
    generation_max_output_tokens: int = 500

    retrieval_max_distance: float = 1.2
    retrieval_min_results: int = 1
    citation_quote_max_chars: int = 180

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()