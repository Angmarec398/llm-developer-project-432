"""Тесты разбора грязных ответов модели: все четыре случая."""
import json

import pytest

from app.schemas.cards import CardSeo, CardSource, ProductCard
from app.services import structured
from app.services.structured import ResponseParseError, extract_json

VALID_CARD_JSON = ProductCard(
    title="Чайник",
    description="Описание",
    seo=CardSeo(slug="chainik"),
    sources=[CardSource(chunk_id=1, document="kettle_spec.xlsx")],
    confidence=0.9,
).model_dump_json()


def test_clean_json_passthrough():
    """Чистый JSON возвращается как есть."""
    assert extract_json('{"title": "Чайник"}') == '{"title": "Чайник"}'


def test_markdown_fence_stripped():
    """JSON в markdown-заборе: забор срезается."""
    raw = '```json\n{"title": "Чайник"}\n```'
    assert extract_json(raw) == '{"title": "Чайник"}'


def test_text_around_json():
    """JSON со вступлением и пояснением: лишний текст отбрасывается."""
    raw = 'Вот карточка:\n{"title": "Чайник"}\nНадеюсь, помог!'
    assert extract_json(raw) == '{"title": "Чайник"}'


def test_garbage_raises():
    """Мусор без JSON: понятная ошибка, а не JSONDecodeError."""
    with pytest.raises(ResponseParseError, match="нет валидного JSON"):
        extract_json("извини, я не смог придумать карточку")


def test_empty_raises():
    """Пустой ответ: отдельная понятная ошибка."""
    with pytest.raises(ResponseParseError, match="пустой ответ"):
        extract_json("   ")


def _mock_answers(monkeypatch, answers: list[str]) -> list[str]:
    """Подменить run_agent очередью ответов; вернуть список промптов."""
    prompts: list[str] = []
    remaining = iter(answers)

    async def fake_run_agent(agent, prompt):
        prompts.append(prompt)
        return next(remaining)

    monkeypatch.setattr(structured, "run_agent", fake_run_agent)
    return prompts


async def test_healing_retries_with_error_in_prompt(monkeypatch):
    """Невалидный JSON: повторный запрос содержит текст ошибки."""
    prompts = _mock_answers(monkeypatch, ["это не json", VALID_CARD_JSON])
    card = await structured.call_validated(None, "сделай карточку", ProductCard)
    assert card.title == "Чайник"
    assert len(prompts) == 2
    assert "не прошёл проверку" in prompts[1]
    assert "это не json" in prompts[1]


async def test_empty_response_raises(monkeypatch):
    """Пустой ответ: после исчерпания попыток — понятная ошибка."""
    _mock_answers(monkeypatch, ["", ""])
    with pytest.raises(ResponseParseError, match="пустой ответ"):
        await structured.call_validated(None, "сделай карточку", ProductCard)


async def test_repair_field_fixes_only_title(monkeypatch):
    """Точечная переделка: заголовок исправлен, описание не тронуто."""
    card = ProductCard(
        title="Чайник",
        description="Нержавеющая сталь, 2200 Вт, объём 1.7 литра.",
        seo=CardSeo(slug="chainik"),
        sources=[CardSource(chunk_id=1, document="kettle_spec.xlsx")],
        confidence=0.9,
    )
    broken = card.model_copy(update={"title": "x" * 61})  # мимо валидаторов
    prompts = _mock_answers(
        monkeypatch, ['{"title": "Чайник стальной 2200 Вт"}'])

    repaired = await structured.repair_field(
        None, broken, "title", "заголовок длиннее 60 символов"
    )

    assert repaired.title == "Чайник стальной 2200 Вт"
    assert repaired.description == card.description
    assert broken.title in prompts[0]  # текущий черновик ушёл в модель
    assert "заголовок длиннее 60 символов" in prompts[0]


async def test_long_title_repaired_pointwise(monkeypatch):
    """Сломано одно поле: точечная переделка без переделки описания."""
    broken_card = json.loads(VALID_CARD_JSON)
    broken_card["title"] = "x" * 61
    prompts = _mock_answers(monkeypatch, [
        json.dumps(broken_card, ensure_ascii=False),
        '{"title": "Чайник стальной"}',
    ])

    card = await structured.call_validated(None, "сделай карточку", ProductCard)

    assert card.title == "Чайник стальной"
    assert card.description == "Описание"  # описание не переделывалось
    assert "Текущий черновик карточки" in prompts[1]  # точечный запрос
    assert "сделай карточку" not in prompts[1]
