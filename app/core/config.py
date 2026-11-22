from decimal import Decimal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen3.5-4b"
    llm_model_utility: str = "llama-3.2-3b-instruct"

    database_url: str = "postgresql://card:card@localhost:5432/card"

    temporal_host: str = "localhost:7233"
    temporal_task_queue: str = "card-queue"

    llm_timeout_s: float = 300.0
    llm_max_retries: int = 3

    validation_max_attempts: int = 2
    card_confidence_threshold: float = 0.6

    upload_dir: str = "data/uploads"
    upload_max_bytes: int = 10 * 1024 * 1024
    upload_allowed_extensions: frozenset[str] = frozenset({".pdf", ".docx", ".xlsx"})

    chunk_size: int = 800
    chunk_overlap: int = 150

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32

    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.80

    context_max_chars: int = 4000

    # Цены за миллион токенов; Decimal, не float — деньги.
    llm_price_input_per_mtok: Decimal = Decimal(0)
    llm_price_output_per_mtok: Decimal = Decimal(0)
    llm_utility_price_input_per_mtok: Decimal = Decimal(0)
    llm_utility_price_output_per_mtok: Decimal = Decimal(0)

    injection_document_threshold: float = 0.5


settings = Settings()
