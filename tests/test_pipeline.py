"""Тесты пайплайна: клиент модели подменён, сеть и LLM не нужны."""
import pytest

from app.schemas.cards import (
    CardSeo,
    CardSource,
    CritiqueReport,
    ProductCard,
    SupplierFacts,
)
from app.services import pipeline, structured

FACTS = SupplierFacts(
    product_name="Чайник", characteristics={"мощность": "2200 Вт"}
)
CARD = ProductCard(
    title="Чайник 2200 Вт",
    description="Описание",
    characteristics={"мощность": "2200 Вт"},
    benefits=["Быстрый"],
    seo=CardSeo(slug="chainik-2200"),
    source=CardSource(document="supplier_text"),
    confidence=0.9,
)


def _mock_run_agent(monkeypatch: pytest.MonkeyPatch, outputs: list) -> None:
    """Подменить run_agent очередью заготовленных ответов модели.

    Патчится в пространстве имён structured: все стадии пайплайна
    зовут run_agent через call_validated.
    """
    answers = iter(outputs)

    async def fake_run_agent(agent, prompt):
        return next(answers)

    monkeypatch.setattr(structured, "run_agent", fake_run_agent)



async def test_pipeline_approved_first_try(monkeypatch: pytest.MonkeyPatch):
    """Успех: критик одобряет первую карточку — один круг, verdict approved."""
    _mock_run_agent(monkeypatch, [FACTS, CARD, CritiqueReport(verdict="approve")])
    card, attempts, verdict = await pipeline.run_pipeline("текст поставщика")
    assert verdict == "approved"
    assert attempts == 1
    assert card == CARD


async def test_pipeline_approved_after_rework(monkeypatch: pytest.MonkeyPatch):
    """Переделка: первая карточка отклонена, исправленная одобрена."""
    improved = CARD.model_copy(update={"title": "Чайник стальной 2200 Вт"})
    _mock_run_agent(monkeypatch, [
        FACTS,
        CARD,
        CritiqueReport(verdict="regenerate", issues=["заголовок длинный"]),
        improved,
        CritiqueReport(verdict="approve"),
    ])
    card, attempts, verdict = await pipeline.run_pipeline("текст поставщика")
    assert verdict == "approved"
    assert attempts == 2
    assert card.title == "Чайник стальной 2200 Вт"


async def test_pipeline_loop_exhausted(monkeypatch: pytest.MonkeyPatch):
    """Петля исчерпана: критик всегда против — стоп после трёх кругов."""
    _mock_run_agent(monkeypatch, [
        FACTS,
        CARD, CritiqueReport(verdict="regenerate", issues=["плохо"]),
        CARD, CritiqueReport(verdict="regenerate", issues=["всё ещё плохо"]),
        CARD, CritiqueReport(verdict="regenerate", issues=["нет"]),
    ])
    _card, attempts, verdict = await pipeline.run_pipeline("текст поставщика")
    assert verdict == "rejected"
    assert attempts == 3


async def test_pipeline_low_confidence_goes_to_human(
    monkeypatch: pytest.MonkeyPatch,
):
    """Критик доволен, но уверенность ниже порога — не «готово», а человек."""
    scarce_card = CARD.model_copy(update={
        "confidence": 0.3,
        "missing_fields": ["гарантия", "цвет"],
    })
    _mock_run_agent(monkeypatch, [
        FACTS, scarce_card, CritiqueReport(verdict="approve"),
    ])
    card, _attempts, verdict = await pipeline.run_pipeline("скудный текст")
    assert verdict == "low_confidence"
    assert card.missing_fields  # непустой список недостающих полей
    assert card.confidence < 0.6
