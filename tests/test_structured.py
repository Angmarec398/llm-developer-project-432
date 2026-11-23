"""Тесты разбора и лечения ответов модели: validate_or_retry, точечная
переделка поля, порог уверенности."""
import asyncio

import pytest

from app.schemas.cards import CardDraft
from app.services import structured
from app.services.json_utils import LLMJsonError, parse_llm_json
from app.services.structured import (
    CardValidationError,
    confidence_gate,
    extract_json,
    regenerate_field,
    validate_or_retry,
)


def _card_json(**overrides) -> str:
    """Валидный JSON карточки с переопределяемыми полями."""
    card = CardDraft(
        title="МиксерПро 800 — блендер 800 Вт",
        description="Погружной блендер с металлической ножкой.",
        characteristics={"Мощность": "800 Вт"},
        confidence=0.9,
    )
    data = card.model_dump()
    data.update(overrides)
    return CardDraft.model_validate(data).model_dump_json()


def _patch_run_agent(monkeypatch, answers: list) -> list[str]:
    """Подменить вызов модели очередью ответов; вернуть список промптов."""
    prompts: list[str] = []
    queue = iter(answers)

    async def fake_run_agent(agent, prompt):
        prompts.append(prompt)
        return next(queue)

    monkeypatch.setattr(structured, "run_agent", fake_run_agent)
    return prompts


def test_parse_clean_json():
    """Чистый JSON разбирается напрямую."""
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_markdown_fence():
    """JSON в markdown-заборе достаётся целиком."""
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_text_around_json():
    """Болтовня вокруг JSON отсекается."""
    assert parse_llm_json('Вот ответ: {"a": 1} надеюсь, поможет') == {"a": 1}


def test_parse_garbage_raises():
    """Без JSON в ответе — понятная ошибка."""
    with pytest.raises(LLMJsonError):
        parse_llm_json("совсем не json")


def test_extract_json_returns_string():
    """extract_json отдаёт строку, пригодную для json.loads."""
    import json

    assert json.loads(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}


def test_validate_or_retry_heals_garbage(monkeypatch):
    """Мусорный ответ уходит обратно с текстом ошибки, вторая попытка валидна."""
    prompts = _patch_run_agent(monkeypatch, ["не json", _card_json()])
    card = asyncio.run(validate_or_retry("факты"))
    assert card.title.startswith("МиксерПро")
    assert len(prompts) == 2
    assert "невалиден" in prompts[1]


def test_validate_or_retry_gives_up(monkeypatch):
    """Все попытки исчерпаны — CardValidationError."""
    _patch_run_agent(monkeypatch, ["мусор"] * 3)
    with pytest.raises(CardValidationError):
        asyncio.run(validate_or_retry("факты"))


def test_regenerate_field_passes_field_and_draft(monkeypatch):
    """Точечная починка: в промпте названо поле и передан черновик."""
    draft = CardDraft(
        title="МиксерПро 800",
        description="описание",
        confidence=0.9,
    )
    prompts = _patch_run_agent(monkeypatch, [_card_json()])
    fixed = asyncio.run(regenerate_field(draft, "title", "слишком длинный"))
    assert fixed.title == "МиксерПро 800 — блендер 800 Вт"
    assert "'title'" in prompts[0]
    assert "слишком длинный" in prompts[0]
    assert "МиксерПро 800" in prompts[0]


def test_confidence_gate_boundaries():
    """Уверенность ниже порога — needs_review, на пороге и выше — done."""
    low = CardDraft(title="t", description="d", confidence=0.3)
    high = CardDraft(title="t", description="d", confidence=0.9)
    assert confidence_gate(low) == "needs_review"
    assert confidence_gate(high) == "done"
    edge = CardDraft(
        title="t", description="d", confidence=structured.CONFIDENCE_GATE
    )
    assert confidence_gate(edge) == "done"
