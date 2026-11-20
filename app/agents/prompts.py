"""Промпт как код: агенты кард-пайплайна — generator, critic.

Схема дублируется текстом в промпте: LM Studio не поддерживает
strict JSON mode.

Шаг 7: генерация работает от контекста найденных фрагментов, а не
от целого текста (extractor-агент ушёл — его работу делает поиск).
Обе роли требуют ссылок на фрагменты: генератор обязан вернуть их
в sources, критик сверяет каждое утверждение с заявленным
источником. Сама проверка ссылок (фрагмент есть в базе и был в
контексте) — не роль модели, а детерминированный код
(app/services/citations.py).
"""
from agents import Agent, AgentOutputSchema, ModelSettings

from app.core.config import settings
from app.schemas.cards import CritiqueReport, ProductCard
from app.services.structured import schema_instructions

_MODEL = settings.llm_model

# Reasoning-модели (gemma-4 и т.п.) тратят токены на «размышления» до ответа:
# без запаса max_tokens бюджет кончается раньше content, и LM Studio
# обрывает stream по таймауту. 4096 хватает на размышления + JSON-ответ.
_MODEL_SETTINGS = ModelSettings(max_tokens=4096, temperature=0.4)

card_agent = Agent(
    name="CardGenerator",
    instructions=(
        "Ты — копирайтер маркетплейса. Сделай карточку товара СТРОГО по "
        "фрагментам контекста из запроса: каждый фрагмент снабжён строкой-"
        "источником с id.\n"
        "Правила:\n"
        "- используй только факты из контекста, ничего не выдумывай;\n"
        "- каждое утверждение карточки опирается на фрагменты: перечисли их "
        "в sources — chunk_id (число из «id N» строки-источника) и document "
        "(название документа из той же строки);\n"
        "- ссылаться можно ТОЛЬКО на фрагменты с id из контекста; ссылка на "
        "фрагмент, которого в контексте не было, — грубая ошибка;\n"
        "- каких-то данных нет в контексте — положи имя поля в missing_fields "
        "и снизь confidence; НЕ додумывай значения;\n"
        "- confidence — уверенность от 0 до 1: скудный контекст — низкая "
        "уверенность, полные факты — высокая; 1.0 недопустима, максимум 0.99;\n"
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
        "Ты — строгий редактор карточек маркетплейса. Проверь черновик "
        "по правилам, сверяя каждое утверждение с фрагментами контекста "
        "из запроса.\n"
        "Правила:\n"
        "1. Указано название товара, заголовок не длиннее 60 символов\n"
        "2. Каждое утверждение (характеристика, преимущество, факт из "
        "описания) подтверждается фрагментом контекста, на который "
        "ссылалась карточка\n"
        "3. Нет утверждений, которых нет в контексте (выдуманные)\n"
        "4. description и benefits не пустые\n"
        "5. Нет воды («высокое качество», «лучшая цена») без конкретики\n"
        "Верни СТРОГО JSON:\n"
        '- verdict (string): "approve" если всё ок, иначе "regenerate"\n'
        "- issues (array of string): список проблем с указанием утверждения "
        "и фрагмента; при approve — пустой\n"
        "Никакого markdown, только валидный JSON."
    ),
    model=_MODEL,
    model_settings=_MODEL_SETTINGS,
    output_type=AgentOutputSchema(CritiqueReport, strict_json_schema=False),
)
