"""Промпт как код: три агента кард-пайплайна — extractor, generator, critic.

Схема дублируется текстом в промпте: LM Studio не поддерживает
strict JSON mode.
"""
from agents import Agent, AgentOutputSchema, ModelSettings

from app.core.config import settings
from app.schemas.cards import CritiqueReport, ProductCard, SupplierFacts
from app.services.structured import schema_instructions

_MODEL = settings.llm_model

# Reasoning-модели (gemma-4 и т.п.) тратят токены на «размышления» до ответа:
# без запаса max_tokens бюджет кончается раньше content, и LM Studio
# обрывает stream по таймауту. 4096 хватает на размышления + JSON-ответ.
_MODEL_SETTINGS = ModelSettings(max_tokens=4096, temperature=0.4)

extractor_agent = Agent(
    name="Extractor",
    instructions=(
        "Ты — извлекатель фактов. Из текста поставщика выдели структурированные факты.\n"
        "Верни СТРОГО JSON с этими полями и никаких других:\n"
        "- product_name (string): название товара\n"
        "- characteristics (object): пары ключ-значение, ТОЛЬКО из текста\n"
        "- missing_fields (array of string): что важно для карточки, но в тексте нет "
        "(например: гарантия, цвет, страна-производитель)\n"
        "Не выдумывай. Нет данных — поле в missing_fields. "
        "Никакого markdown, только валидный JSON."
    ),
    model=_MODEL,
    model_settings=_MODEL_SETTINGS,
    output_type=AgentOutputSchema(SupplierFacts, strict_json_schema=False),
)

card_agent = Agent(
    name="CardGenerator",
    instructions=(
        "Ты — копирайтер маркетплейса. Из фактов о товаре сделай карточку.\n"
        "Правила:\n"
        "- используй только предоставленные факты, ничего не выдумывай;\n"
        "- каких-то данных нет — положи имя поля в missing_fields "
        "и снизь confidence; НЕ додумывай значения;\n"
        "- confidence — уверенность от 0 до 1: скудный исходный текст — "
        "низкая уверенность, полные факты — высокая;\n"
        "- в source.document укажи имя источника, переданное в запросе;\n"
        "- заголовок не длиннее 60 символов.\n"
        + schema_instructions(ProductCard)
    ),
    model=_MODEL,
    model_settings=_MODEL_SETTINGS,
    output_type=AgentOutputSchema(ProductCard, strict_json_schema=False),
)

critic_agent = Agent(
    name="Critic",
    instructions=(
        "Ты — строгий редактор карточек маркетплейса. Проверь черновик по правилам:\n"
        "1. Указанно название товара или его тип\n"
        "2. title не длиннее 60 символов\n"
        "3. Нет характеристик, которых нет в фактах (выдуманные)\n"
        "4. description и benefits не пустые\n"
        "5. Нет воды («высокое качество», «лучшая цена») без конкретики\n"
        "Верни СТРОГО JSON:\n"
        '- verdict (string): "approve" если всё ок, иначе "regenerate"\n'
        "- issues (array of string): список проблем; при approve — пустой\n"
        "Никакого markdown, только валидный JSON."
    ),
    model=_MODEL,
    model_settings=_MODEL_SETTINGS,
    output_type=AgentOutputSchema(CritiqueReport, strict_json_schema=False),
)
