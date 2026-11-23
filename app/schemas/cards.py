from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SupplierFacts(BaseModel):
    """Результат extractor-агента: структурированные факты из текста поставщика."""

    product_name: str = Field(description="Название товара из текста поставщика")
    characteristics: dict[str, str] = Field(
        default_factory=dict, description="Характеристики: ключ-значение"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Что не найдено в тексте (например: гарантия, цвет)",
    )


class SourceRef(BaseModel):
    """Ссылка на источник утверждения: конкретный чанк документа.

    chunk_id — идентификатор чанка в базе: карточка обязана сослаться
    на чанк из поданного контекста, а приложение потом проверяет, что
    чанк существует и был именно в этом контексте.
    """

    chunk_id: str = Field(description="Идентификатор чанка в БД")
    doc_id: str = Field(default="", description="Идентификатор документа")
    page: int | None = Field(default=None, description="Страница источника")
    score: float | None = Field(default=None, description="Релевантность чанка")


class Seo(BaseModel):
    """SEO-блок карточки."""

    title: str = Field(default="", description="SEO-заголовок до 100 символов")
    description: str = Field(default="", description="SEO-описание до 200 символов")
    keywords: list[str] = Field(default_factory=list)


class CardDraft(BaseModel):
    """Результат generator-агента: черновик карточки товара.

    Полный контракт карточки: контент-поля, SEO-блок, цитирование
    источников, честность о неполноте данных и самооценка уверенности.
    """

    title: str = Field(description="Заголовок карточки, до 60 символов")
    description: str = Field(description="Описание на 3-4 предложения")
    characteristics: dict[str, str] = Field(default_factory=dict)
    benefits: list[str] = Field(default_factory=list)
    seo: Seo = Field(default_factory=Seo)
    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Источники утверждений: chunk_id + doc_id + page",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Что важно для карточки, но данных нет; поле НЕ выдумывать",
    )
    confidence: float = Field(
        default=0.5,
        description="Самооценка модели: 0.0 — данных нет, 1.0 — всё подтверждено",
    )

    @field_validator("title")
    @classmethod
    def title_max_length(cls, value: str) -> str:
        """Заголовок длиннее 60 символов отклоняется с явной ошибкой."""
        if len(value) > 60:
            raise ValueError(
                f"заголовок длиннее 60 символов ({len(value)}): сократи"
            )
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, value: float) -> float:
        """Уверенность — число от нуля до единицы."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence должен быть в [0.0, 1.0]")
        return value

    @field_validator("characteristics")
    @classmethod
    def no_empty_characteristics(cls, value: dict[str, str]) -> dict[str, str]:
        """Характеристики с пустыми значениями молча выбрасываются."""
        return {
            key: characteristic
            for key, characteristic in value.items()
            if str(characteristic).strip()
        }

    @model_validator(mode="after")
    def missing_fields_not_in_characteristics(self) -> "CardDraft":
        """Ключ не может быть одновременно в characteristics и missing_fields."""
        duplicates = set(self.missing_fields) & set(self.characteristics.keys())
        if duplicates:
            raise ValueError(
                f"поля одновременно в characteristics и missing_fields: "
                f"{sorted(duplicates)}"
            )
        return self


class CritiqueReport(BaseModel):
    """Результат critic-агента: вердикт и список проблем."""

    verdict: Literal["approve", "regenerate"] = Field(
        description="approve — карточка годится; regenerate — переделать"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Список проблем: заголовок длиннее 60 символов, "
        "выдуманные характеристики, пустые поля и т.п.",
    )

    @field_validator("issues", mode="before")
    @classmethod
    def _coerce_issue_items(cls, value: list) -> list[str]:
        """Критик слабой служебной моделью отдаёт объекты вместо строк.

        Встреченная форма: {"assertion": "..."}. Приводим к строке —
        берём значение поля assertion, иначе первое значение объекта:
        замечание не теряется, разбор не падает, починка не нужна.
        """
        coerced: list[str] = []
        for item in value or []:
            if isinstance(item, dict):
                text = item.get("assertion") or next(iter(item.values()), "")
                coerced.append(str(text))
            else:
                coerced.append(item)
        return coerced


class JobIn(BaseModel):
    """Запрос на генерацию карточки (шаг 7): документы + подсказка."""

    document_ids: list[str] = Field(min_length=1)
    hint: str | None = Field(
        default=None,
        description="Подсказка, что это за товар (название/тип); "
        "без неё запрос строится по заголовкам документов",
    )


class FaithfulnessVerdict(BaseModel):
    """Вердикт модели-судьи (шаг 8): какие утверждения карточки
    подтверждаются чанками-источниками, а каких в них нет.

    Судья только размечает; долю считает детерминированный код
    (evals/metrics.py) — оценке модели доверяем как оценке,
    а не как истине.
    """

    supported: list[str] = Field(
        default_factory=list,
        description="Утверждения карточки, прямо подтверждённые чанками",
    )
    unsupported: list[str] = Field(
        default_factory=list,
        description="Утверждения, которых в чанках нет",
    )


class MaskedItem(BaseModel):
    """Замаскированное ПДн в отчёте защиты (шаг 9): категория и метка.

    Сырое значение в отчёт не попадает — только метка, которой оно
    заменено в тексте: отчёт безопасно кладётся в логи, базу и
    результат задачи.
    """

    category: str      # phone | email | taxpayer_number | card_number
    label: str         # метка-заглушка, подставленная в текст


class ExcludedChunk(BaseModel):
    """Чанк, исключённый из контекста детектором инъекций."""

    chunk_id: str
    doc_id: str
    reasons: list[str] = Field(default_factory=list)


class QuarantinedDocument(BaseModel):
    """Документ, отозванный к человеку: подозрительных чанков
    больше порога (настройка suspicious_chunk_limit)."""

    doc_id: str
    reason: str


class GuardrailReport(BaseModel):
    """Отчёт защитных рубежей (шаг 9) — часть результата задачи.

    Что замаскировано на входе (masked), какие чанки исключены
    (excluded_chunks), какие документы в карантине (quarantined_
    documents), что замаскировано в готовой карточке (output_masked).
    Только метки и причины, без сырых значений ПДн.
    """

    masked: list[MaskedItem] = Field(default_factory=list)
    excluded_chunks: list[ExcludedChunk] = Field(default_factory=list)
    quarantined_documents: list[QuarantinedDocument] = Field(
        default_factory=list
    )
    output_masked: list[MaskedItem] = Field(default_factory=list)
