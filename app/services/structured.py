"""Валидация ответа LLM по строгому контракту + лечение невалидных ответов.

Три уровня надёжности structured output:
1. parse_llm_json — достаём JSON из «грязного» ответа модели.
2. validate_or_retry — при ошибке шлём её текст обратно модели
   (число исправлений ограничено MAX_FIX_ATTEMPTS).
3. regenerate_field — чиним одно сломанное поле, не перегенерируя карточку.

Схема дублируется текстом в промпте (schema_instructions): LM Studio не
поддерживает strict JSON mode, а грамматика structured output обнуляет
свободные dict (ADR-0004), поэтому output_type агентам не задаётся.
"""
import json

from pydantic import BaseModel, ValidationError

from app.agents.prompts import fixer_agent, generator_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import run_agent
from app.schemas.cards import CardDraft
from app.services.json_utils import LLMJsonError, parse_llm_json

log = get_logger(__name__)

MAX_FIX_ATTEMPTS = 2
CONFIDENCE_GATE = settings.card_confidence_threshold

# Старое имя ошибки разбора — псевдоним канонического LLMJsonError.
ResponseParseError = LLMJsonError


class CardValidationError(Exception):
    """Карточку не удалось валидировать за отведённые попытки."""


def extract_json(raw_text: str) -> str:
    """Вытащить JSON-строку из грязного ответа модели.

    Возвращает строку, пригодную для json.loads / model_validate_json.
    Бросает LLMJsonError (он же ResponseParseError), если JSON найти
    не удалось.
    """
    return json.dumps(parse_llm_json(raw_text), ensure_ascii=False)


def schema_instructions(output_model: type[BaseModel]) -> str:
    """Текстовая копия JSON-схемы для инструкции агента.

    Схема в запрос не передаётся (output_type у агентов убран: грамматика
    LM Studio обнуляет свободные dict) — модель видит форму ответа только
    текстом, грязный JSON разбирает parse_llm_json.
    """
    schema_json = json.dumps(
        output_model.model_json_schema(), ensure_ascii=False, indent=2
    )
    return "Верни СТРОГО JSON по этой схеме:\n```json\n" + schema_json + "\n```"


async def validate_or_retry(prompt: str, feedback_prefix: str = "") -> CardDraft:
    """Сгенерировать карточку и провалидировать; при ошибке — повтор с текстом ошибки.

    feedback_prefix — замечания критика (если это регенерация пайплайна).
    Бросает CardValidationError после MAX_FIX_ATTEMPTS неудачных исправлений.
    """
    current_prompt = feedback_prefix + prompt
    last_error: Exception | None = None
    for attempt in range(MAX_FIX_ATTEMPTS + 1):
        raw_output = await run_agent(generator_agent, current_prompt)
        try:
            data = parse_llm_json(raw_output)
            return CardDraft.model_validate(data)
        except (LLMJsonError, ValidationError) as error:
            last_error = error
            log.warning(
                "card_validation_failed", attempt=attempt + 1, error=str(error)[:300]
            )
            current_prompt = (
                f"Твой предыдущий ответ невалиден. Ошибка валидации:\n{error}\n\n"
                f"Исправь и верни СТРОГО валидный JSON карточки по контракту. "
                f"Предыдущий запрос:\n{current_prompt}"
            )
    raise CardValidationError(f"карточка не прошла валидацию: {last_error}")


async def regenerate_field(draft: CardDraft, field: str, error: str) -> CardDraft:
    """Точечная регенерация: чиним одно поле, остальное остаётся как есть.

    Дешевле полной регенерации: модель получает текущий черновик
    и конкретную ошибку.
    """
    prompt = (
        "В черновике карточки поле "
        f"{field!r} не прошло валидацию:\n{error}\n\n"
        "Текущий черновик (JSON):\n"
        + draft.model_dump_json(indent=2)
        + "\n\nПерегенерируй ТОЛЬКО это поле, остальное верни без изменений. "
        "Ответ — полный JSON карточки."
    )
    raw_output = await run_agent(fixer_agent, prompt)
    data = parse_llm_json(raw_output)
    return CardDraft.model_validate(data)


def confidence_gate(draft: CardDraft) -> str:
    """Низкая уверенность → карточка требует ручной проверки."""
    return "needs_review" if draft.confidence < CONFIDENCE_GATE else "done"
