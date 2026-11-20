"""Тесты стадий пайплайна: генерация от контекста, критик сверяет с источниками.

Клиент модели подменён, сеть и LLM не нужны. Цикл перегенерации
тестируется на уровне оркестрации в сквозном тесте (tests/test_end_to_end.py).
"""
import pytest

from app.schemas.cards import CardSeo, CardSource, CritiqueReport, ProductCard
from app.services import pipeline, structured

CONTEXT = "[1] документ doc-1, фрагмент 1, id 7\nМощность 2200 Вт"
CARD = ProductCard(
    title="Чайник 2200 Вт",
    description="Описание",
    characteristics={"мощность": "2200 Вт"},
    benefits=["Быстрый"],
    seo=CardSeo(slug="chainik-2200"),
    sources=[CardSource(chunk_id=7, document="doc-1")],
    confidence=0.9,
)


def _mock_run_agent(monkeypatch: pytest.MonkeyPatch, outputs: list) -> list[str]:
    """Подменить run_agent очередью ответов; вернуть список промптов.

    Патчится в пространстве имён structured: обе стадии зовут
    run_agent через call_validated.
    """
    prompts: list[str] = []
    answers = iter(outputs)

    async def fake_run_agent(agent, prompt):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(structured, "run_agent", fake_run_agent)
    return prompts


async def test_generate_sends_context_to_model(monkeypatch: pytest.MonkeyPatch):
    """Генерация: промпт содержит контекст с опознаваемым фрагментом."""
    prompts = _mock_run_agent(monkeypatch, [CARD])
    card = await pipeline.generate_card(CONTEXT)
    assert card == CARD
    assert CONTEXT in prompts[0]
    assert "id 7" in prompts[0]


async def test_generate_includes_feedback(monkeypatch: pytest.MonkeyPatch):
    """Переделка: замечания (нарушения ссылок, критик) уходят в промпт."""
    prompts = _mock_run_agent(monkeypatch, [CARD])
    await pipeline.generate_card(
        CONTEXT, feedback=["фрагмент 99 не существует в базе"]
    )
    assert "Замечания к переделке" in prompts[0]
    assert "фрагмент 99 не существует в базе" in prompts[0]


async def test_critique_gets_context_and_card(monkeypatch: pytest.MonkeyPatch):
    """Критик получает и контекст, и карточку: сверяет утверждения с источниками."""
    prompts = _mock_run_agent(monkeypatch, [CritiqueReport(verdict="approve")])
    report = await pipeline.critique(CONTEXT, CARD)
    assert report.verdict == "approve"
    assert CONTEXT in prompts[0]
    assert "Чайник 2200 Вт" in prompts[0]
