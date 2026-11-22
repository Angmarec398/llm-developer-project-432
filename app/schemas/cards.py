from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TITLE_MAX_LENGTH = 60


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


class CardSeo(BaseModel):
    """SEO-блок карточки."""
    slug: str          # URL-часть
    keywords: list[str] = Field(default_factory=list)


class CardSource(BaseModel):
    """Ссылка на фрагмент-источник данных карточки.

    chunk_id — идентификатор фрагмента в базе: карточка обязана
    сослаться на фрагмент из поданного контекста, а приложение
    потом проверяет, что фрагмент существует и был именно в этом
    контексте. document — читаемое название документа для человека.
    """

    chunk_id: int      # id фрагмента в базе — по нему проверяется ссылка
    document: str      # название/идентификатор документа (читаемое)
    url: str | None = None


class ProductCard(BaseModel):
    """Финальная карточка товара — контракт шагов 4 и 7.

    Каждое утверждение карточки опирается на фрагменты-источники:
    модель обязана вернуть их в sources, проверка ссылок гарантирует,
    что фрагменты существуют и были видны модели в контексте.
    """
    title: str = Field(max_length=TITLE_MAX_LENGTH)
    description: str
    characteristics: dict[str, str] = Field(default_factory=dict)
    benefits: list[str] = Field(default_factory=list)
    seo: CardSeo
    sources: list[CardSource] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)

    @field_validator("characteristics")
    @classmethod
    def _no_empty_values(cls, value: dict[str, str]) -> dict[str, str]:
        """Отвергнуть характеристики с пустыми значениями."""
        empty_keys = [
            key for key, characteristic_value in value.items()
            if not characteristic_value.strip()
        ]
        if empty_keys:
            raise ValueError(
                f"пустые значения у характеристик: {', '.join(empty_keys)}"
            )
        return value

    @model_validator(mode="after")
    def _not_in_both_lists(self) -> "ProductCard":
        """Ключ не может быть одновременно в characteristics и missing_fields."""
        both = set(self.characteristics) & set(self.missing_fields)
        if both:
            raise ValueError(
                "поля и в характеристиках, и в недостающих: "
                + ", ".join(sorted(both))
            )
        return self


class FaithfulnessVerdict(BaseModel):
    """Вердикт модели-судьи (шаг 8): какие утверждения карточки
    подтверждаются фрагментами-источниками, а каких в них нет.

    Судья только размечает; долю считает детерминированный код
    (evals/metrics.py) — оценке модели доверяем как оценке,
    а не как истине.
    """

    supported: list[str] = Field(
        default_factory=list,
        description="Утверждения карточки, прямо подтверждённые фрагментами",
    )
    unsupported: list[str] = Field(
        default_factory=list,
        description="Утверждения, которых во фрагментах нет",
    )

class InjectionVerdict(BaseModel):
    """Вердикт точного уровня детектора инъекций (шаг 9): содержит ли
    фрагмент попытку манипуляции инструкциями ИИ. Заполняет дешёвая
    служебная модель; решает только второй уровень, первый — правила
    (app/guardrails/injection.py), которые и решают, идти ли в модель.
    """

    is_injection: bool = Field(
        description="true — фрагмент содержит попытку манипуляции "
        "инструкциями ИИ"
    )
    reason: str = Field(
        default="",
        description="что именно подозрительного найдено во фрагменте",
    )

class MaskedItem(BaseModel):
    """Замаскированное ПДн в отчёте защиты (шаг 9): категория и метка.

    Сырое значение в отчёт не попадает — только метка, которой оно
    заменено в тексте: отчёт безопасно кладётся в логи, базу и
    результат задачи.
    """

    category: str      # phone | email | taxpayer_number | card_number
    label: str         # «телефон номер один»


class ExcludedChunk(BaseModel):
    """Фрагмент, исключённый из контекста детектором инъекций."""

    chunk_id: int
    document_id: str
    reasons: list[str] = Field(default_factory=list)


class QuarantinedDocument(BaseModel):
    """Документ, отозванный к человеку: подозрительных фрагментов
    больше порога (настройка injection_document_threshold)."""

    document_id: str
    reason: str


class GuardrailReport(BaseModel):
    """Отчёт защитных рубежей (шаг 9) — часть результата задачи.

    Что замаскировано на входе (masked), какие фрагменты исключены
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
