from decimal import Decimal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    # Имя — как в списке /v1/models LM Studio: у моделей из HF-префикса
    # (unsloth/...) сервер отдаёт имя без него, а Agents SDK распарсил бы
    # «префикс/модель» как неизвестного провайдера (UserError: Unknown prefix).
    llm_model: str = "qwen3-4b-instruct-2507"
    llm_model_utility: str = "llama-3.2-3b-instruct"

    database_url: str = "postgresql://card:card@localhost:5432/card"

    temporal_host: str = "localhost:7233"
    temporal_task_queue: str = "card-queue"

    llm_timeout_s: float = 300.0
    llm_max_retries: int = 3

    # Порог уверенности: ниже — карточка уходит на ручную проверку.
    card_confidence_threshold: float = 0.5

    upload_dir: str = "data/uploads"
    upload_max_bytes: int = 10 * 1024 * 1024
    upload_allowed_extensions: frozenset[str] = frozenset({".pdf", ".docx", ".xlsx"})

    embed_model: str = "google/embeddinggemma-300m"
    # Размерность векторов модели; колонка chunks.embedding создаётся
    # миграциями под неё (vector(embed_dim)).
    embed_dim: int = 768
    # cpu по умолчанию: на Apple Metal (mps) PyTorch падает в дочерних процессах.
    embed_device: str = "cpu"

    retrieval_top_k: int = 5
    # embeddinggemma-300m даёт косинус 0.4-0.55 для релевантных пар:
    # порог выше режет нормальные результаты, ниже — пускает мусор.
    retrieval_threshold: float = 0.35

    # Цены за миллион токенов; Decimal, не float — деньги.
    llm_price_input_per_mtok: Decimal = Decimal(0)
    llm_price_output_per_mtok: Decimal = Decimal(0)
    llm_utility_price_input_per_mtok: Decimal = Decimal(0)
    llm_utility_price_output_per_mtok: Decimal = Decimal(0)

    injection_document_threshold: float = 0.5
    # Доля подозрительных фрагментов в документе, при которой он уходит
    # на ручную проверку целиком (см. app/guardrails).
    suspicious_chunk_limit: int = 2


settings = Settings()
