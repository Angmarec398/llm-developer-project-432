"""Тесты стадий пайплайна с подменой модели: extract → generate → critique."""
import asyncio

from app.schemas.cards import CardDraft, CritiqueReport, SupplierFacts
from app.services import pipeline
from app.services import structured as structured_mod


def _facts() -> SupplierFacts:
    """Факты о блендере для сценариев."""
    return SupplierFacts(
        product_name="Блендер погружной «МиксерПро 800»",
        characteristics={"Мощность": "800 Вт", "Питание": "220 В"},
    )


def _draft(confidence: float = 0.9) -> CardDraft:
    """Черновик карточки для сценариев."""
    return CardDraft(
        title="МиксерПро 800 — блендер 800 Вт",
        description="Погружной блендер с металлической ножкой.",
        characteristics={"Мощность": "800 Вт"},
        benefits=["Металлическая ножка"],
        confidence=confidence,
    )


def _mock_llm(monkeypatch, answers: list) -> list[str]:
    """Подменить вызовы модели очередью ответов (оба модуля-потребителя)."""
    prompts: list[str] = []
    queue = iter(answers)

    async def fake_run_agent(agent, prompt):
        prompts.append(prompt)
        result = next(queue)
        return result if isinstance(result, str) else result.model_dump_json()

    monkeypatch.setattr(pipeline, "run_agent", fake_run_agent)
    monkeypatch.setattr(structured_mod, "run_agent", fake_run_agent)
    return prompts


def test_extract_parses_facts(monkeypatch):
    """Extractor: факты из JSON модели, даже в markdown-заборе."""
    _mock_llm(monkeypatch, ["```json\n" + _facts().model_dump_json() + "\n```"])
    facts = asyncio.run(pipeline.extract("текст поставщика"))
    assert facts.product_name.startswith("Блендер")
    assert facts.characteristics["Мощность"] == "800 Вт"


def test_pipeline_approved_on_first_try(monkeypatch):
    """Критик доволен с первого раза: одна генерация, статус done."""
    _mock_llm(monkeypatch, [_facts(), _draft(), CritiqueReport(verdict="approve")])
    draft, attempts, verdict, status = asyncio.run(pipeline.run_pipeline("текст"))
    assert verdict == "approved"
    assert status == "done"
    assert attempts == 1
    assert draft.title.startswith("МиксерПро")


def test_pipeline_regenerates_on_critique(monkeypatch):
    """Замечания критика уходят в следующую генерацию как feedback."""
    prompts = _mock_llm(monkeypatch, [
        _facts(),
        _draft(), CritiqueReport(verdict="regenerate", issues=["много воды"]),
        _draft(), CritiqueReport(verdict="approve"),
    ])
    draft, attempts, verdict, status = asyncio.run(pipeline.run_pipeline("текст"))
    assert verdict == "approved"
    assert attempts == 2
    # вызовы: extract, generate#1, critique#1, generate#2 — feedback в четвёртом
    assert "много воды" in prompts[3]


def test_pipeline_rejected_after_max_attempts(monkeypatch):
    """Петля генераций исчерпана — вердикт rejected."""
    _mock_llm(monkeypatch, [
        _facts(),
        _draft(), CritiqueReport(verdict="regenerate", issues=["плохо"]),
        _draft(), CritiqueReport(verdict="regenerate", issues=["всё ещё плохо"]),
    ])
    _draft_out, attempts, verdict, status = asyncio.run(
        pipeline.run_pipeline("текст", max_attempts=2)
    )
    assert verdict == "rejected"
    assert status == "rejected"
    assert attempts == 2


def test_pipeline_low_confidence_needs_review(monkeypatch):
    """Одобренная критиком карточка с низкой уверенностью — к человеку."""
    _mock_llm(monkeypatch, [
        _facts(), _draft(confidence=0.2), CritiqueReport(verdict="approve"),
    ])
    _draft_out, _attempts, verdict, status = asyncio.run(
        pipeline.run_pipeline("текст")
    )
    assert verdict == "approved"
    assert status == "needs_review"
