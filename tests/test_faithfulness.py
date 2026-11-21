"""Тест третьей метрики: вердикт судьи на подмене LLM и расчёт доли.

Сеть и LM Studio не нужны: run_agent в structured.py подменяется,
«ответ модели» — фиксированный JSON.
"""
import json

from app.schemas.cards import (
    CardSeo,
    CardSource,
    FaithfulnessVerdict,
    ProductCard,
)
from app.services import pipeline, structured
from evals.metrics import faithfulness_score


def _minimal_card() -> ProductCard:
    """Минимально валидная карточка для вызова судьи."""
    return ProductCard(
        title="Блендер МиксерПро 800",
        description="Блендер мощностью 800 Вт со стаканом 1,5 л.",
        characteristics={"Мощность": "800 Вт"},
        benefits=["Мерный стакан 1,5 л в комплекте"],
        seo=CardSeo(slug="blender-miksipro-800"),
        sources=[CardSource(chunk_id=1, document="паспорт")],
        confidence=0.9,
    )


async def test_judge_returns_verdict_on_stubbed_llm(monkeypatch):
    """Подменённый LLM отдаёт JSON вердикта — call_validated парсит его в модель."""

    async def fake_run_agent(agent, prompt):
        return json.dumps(
            {"supported": ["Мощность 800 Вт"], "unsupported": ["Вес 1,2 кг"]},
            ensure_ascii=False,
        )

    monkeypatch.setattr(structured, "run_agent", fake_run_agent)
    verdict = await pipeline.judge_faithfulness("контекст", _minimal_card())
    assert isinstance(verdict, FaithfulnessVerdict)
    assert verdict.supported == ["Мощность 800 Вт"]
    assert verdict.unsupported == ["Вес 1,2 кг"]


def test_faithfulness_score_is_supported_share():
    """Доля подтверждённых: половина — 0.5; пустой вердикт — 1.0."""
    half = FaithfulnessVerdict(supported=["a"], unsupported=["b"])
    assert faithfulness_score(half) == 0.5
    empty = FaithfulnessVerdict(supported=[], unsupported=[])
    assert faithfulness_score(empty) == 1.0
