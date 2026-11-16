"""Тесты пайплайна: клиент модели подменён, сеть и LLM не нужны."""
import pytest

from app.schemas.cards import CardDraft, CritiqueReport, SupplierFacts
from app.services import pipeline

FACTS = SupplierFacts(
    product_name="Чайник", characteristics={"мощность": "2200 Вт"}
)
DRAFT = CardDraft(
    title="Чайник 2200 Вт",
    description="Описание",
    characteristics={"мощность": "2200 Вт"},
    benefits=["Быстрый"],
)


def _mock_run_agent(monkeypatch: pytest.MonkeyPatch, outputs: list) -> None:
    """Подменить run_agent очередью заготовленных ответов модели."""
    answers = iter(outputs)

    async def fake_run_agent(agent, prompt):
        return next(answers)

    monkeypatch.setattr(pipeline, "run_agent", fake_run_agent)


async def test_pipeline_approved_first_try(monkeypatch: pytest.MonkeyPatch):
    """Успех: критик одобряет первый черновик — один круг, verdict approved."""
    _mock_run_agent(monkeypatch, [FACTS, DRAFT, CritiqueReport(verdict="approve")])
    draft, attempts, verdict = await pipeline.run_pipeline("текст поставщика")
    assert verdict == "approved"
    assert attempts == 1
    assert draft == DRAFT


async def test_pipeline_approved_after_rework(monkeypatch: pytest.MonkeyPatch):
    """Переделка: первый черновик отклонён, исправленный одобрен."""
    improved = DRAFT.model_copy(update={"title": "Чайник стальной 2200 Вт"})
    _mock_run_agent(monkeypatch, [
        FACTS,
        DRAFT,
        CritiqueReport(verdict="regenerate", issues=["заголовок длинный"]),
        improved,
        CritiqueReport(verdict="approve"),
    ])
    draft, attempts, verdict = await pipeline.run_pipeline("текст поставщика")
    assert verdict == "approved"
    assert attempts == 2
    assert draft.title == "Чайник стальной 2200 Вт"


async def test_pipeline_loop_exhausted(monkeypatch: pytest.MonkeyPatch):
    """Петля исчерпана: критик всегда против — стоп после трёх кругов."""
    _mock_run_agent(monkeypatch, [
        FACTS,
        DRAFT, CritiqueReport(verdict="regenerate", issues=["плохо"]),
        DRAFT, CritiqueReport(verdict="regenerate", issues=["всё ещё плохо"]),
        DRAFT, CritiqueReport(verdict="regenerate", issues=["нет"]),
    ])
    _draft, attempts, verdict = await pipeline.run_pipeline("текст поставщика")
    assert verdict == "rejected"
    assert attempts == 3
