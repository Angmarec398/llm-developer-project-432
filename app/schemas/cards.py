from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TITLE_MAX_LENGTH = 60


class SupplierFacts(BaseModel):
    """Результат extractor-агента: структурированные факты из текста поставщика."""

    product_name: str = Field(
        description="Название товара из текста поставщика")
    characteristics: dict[str, str] = Field(
        default_factory=dict, description="Характеристики: ключ-значение"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Что не найдено в тексте (например: гарантия, цвет)",
    )


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


class JobIn(BaseModel):
    supplier_text: str


class CardSeo(BaseModel):
    """SEO-блок карточки."""
    slug: str          # URL-часть
    keywords: list[str] = Field(default_factory=list)


class CardSource(BaseModel):
    """Ссылка на источник данных карточки."""
    document: str      # имя файла/идентификатор источника
    url: str | None = None


class ProductCard(BaseModel):
    """Финальная карточка товара — контракт шага 4."""
    title: str = Field(max_length=TITLE_MAX_LENGTH)
    description: str
    characteristics: dict[str, str] = Field(default_factory=dict)
    benefits: list[str] = Field(default_factory=list)
    seo: CardSeo
    source: CardSource
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
