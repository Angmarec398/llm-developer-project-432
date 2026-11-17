"""Разбор ответов модели: ответ — это строка, и она грязная.

Четыре случая, которые разбор обязан пережить:
1. чистый JSON;
2. JSON в markdown-заборе (```json ... ```);
3. JSON с текстом до и/или после («Вот ответ: {...} Надеюсь, помог»);
4. мусор или пустая строка — понятная ошибка, а не падение.
"""
import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import run_agent

log = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ResponseParseError(ValueError):
    """Ответ модели не содержит разбираемого JSON."""


def extract_json(raw_text: str) -> str:
    """Вытащить JSON-строку из грязного ответа модели.

    Возвращает строку, пригодную для json.loads / model_validate_json.
    Бросает ResponseParseError, если JSON найти не удалось.
    """
    cleaned = (raw_text or "").strip()
    if not cleaned:
        raise ResponseParseError("пустой ответ модели")

    candidates: list[str] = []
    fence_match = _FENCE_PATTERN.search(cleaned)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(cleaned)
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(cleaned[brace_start:brace_end + 1])

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    raise ResponseParseError(
        f"в ответе модели нет валидного JSON: {cleaned[:120]!r}"
    )


def schema_instructions(output_model: type[BaseModel]) -> str:
    """Текстовая копия JSON-схемы для инструкции агента.

    SDK передаёт схему в запрос через output_type (response_format),
    но LM Studio не поддерживает strict JSON mode, поэтому схема
    дублируется текстом — иначе модель не знает форму ответа.
    """
    schema_json = json.dumps(
        output_model.model_json_schema(), ensure_ascii=False, indent=2
    )
    return (
        "Верни СТРОГО JSON по этой схеме:\n```json\n"
        + schema_json
        + "\n```"
    )


async def call_validated(
    agent,
    prompt: str,
    output_model: type[ModelT],
    max_attempts: int | None = None,
) -> ModelT:
    """Вызвать модель и вернуть валидированный объект.

    Невалидный ответ не выбрасывается: если сломано одно поле —
    точечная переделка (repair_field), иначе ответ уходит обратно
    в модель с текстом ошибки. Число попыток ограничено настройкой
    validation_max_attempts, после исчерпания — последняя ошибка.
    """
    attempts_limit = max_attempts or settings.validation_max_attempts
    current_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(1, attempts_limit + 1):
        raw_output = await run_agent(agent, current_prompt)
        if isinstance(raw_output, output_model):
            return raw_output
        json_text = ""
        try:
            json_text = extract_json(str(raw_output))
            return output_model.model_validate_json(json_text)
        except (ResponseParseError, ValidationError) as validation_error:
            last_error = validation_error
            if isinstance(validation_error, ValidationError) and json_text:
                broken_fields = {
                    error_info["loc"][0]
                    for error_info in validation_error.errors()
                    if error_info["loc"]
                }
                if len(broken_fields) == 1:
                    repaired = await _try_repair_field(
                        agent, output_model, json_text,
                        str(broken_fields.pop()), validation_error,
                    )
                    if repaired is not None:
                        return repaired
            log.info(
                "validation_healing_retry",
                attempt=attempt,
                error=str(validation_error)[:200],
            )
            current_prompt = _repair_prompt(
                prompt, str(raw_output), validation_error
            )
    raise last_error


async def _try_repair_field(
    agent,
    output_model: type[ModelT],
    json_text: str,
    field_name: str,
    error: ValidationError,
) -> ModelT | None:
    """Попробовать точечно исправить единственное битое поле.

    Черновик собирается в обход валидаторов (model_construct) — битый
    объект иначе не создать. Возвращает None, если переделка не вышла:
    тогда сработает обычное лечение полным повторным запросом.
    """
    draft = output_model.model_construct(**json.loads(json_text))
    try:
        repaired_card = await repair_field(
            agent, draft, field_name, str(error)
        )
        return repaired_card  # type: ignore[return-value]
    except (ResponseParseError, ValidationError, json.JSONDecodeError) as repair_error:
        log.info(
            "field_repair_failed",
            field=field_name,
            error=str(repair_error)[:200],
        )
        return None


def _repair_prompt(
    original_prompt: str, bad_output: str, error: Exception
) -> str:
    """Промпт лечения: исходная задача + битый ответ + текст ошибки."""
    return (
        original_prompt
        + "\n\nТвой предыдущий ответ не прошёл проверку."
        + f"\nОтвет:\n{bad_output[:500]}"
        + f"\nОшибка: {error}"
        + "\nИсправь ошибку и верни ТОЛЬКО валидный JSON, без markdown."
    )


async def repair_field(
    agent,
    card: BaseModel,
    field_name: str,
    issue: str,
) -> BaseModel:
    """Точечно переделать одно поле карточки, не трогая остальные.

    Модель получает текущий черновик и текст проблемы, возвращает
    JSON с одним исправленным полем. Остальные поля берутся из
    черновика без изменений. Результат проходит полную валидацию
    схемой — исправление одного поля не должно ломать инварианты.

    Черновик может быть собран через model_construct (в обход
    валидаторов), поэтому warnings сериализатора отключены: финальная
    model_validate всё равно проверит и приведёт каждое поле.
    """
    repair_prompt = (
        "Текущий черновик карточки:\n"
        + card.model_dump_json(indent=2, warnings=False)
        + f"\n\nПоле «{field_name}» не прошло проверку: {issue}"
        + f'\nВерни СТРОГО JSON с одним полем: {{"{field_name}": <исправленное значение>}}'
        + "\nНикакого markdown, только валидный JSON."
    )
    log.info("field_repair_start", field=field_name, issue=issue[:120])
    raw_output = await run_agent(agent, repair_prompt)
    patch = json.loads(extract_json(str(raw_output)))
    if field_name not in patch:
        raise ResponseParseError(
            f"модель не вернула исправленное поле «{field_name}»: "
            f"{str(raw_output)[:120]!r}"
        )
    patched_card = card.model_copy(update={field_name: patch[field_name]})
    return type(card).model_validate(patched_card.model_dump(warnings=False))
