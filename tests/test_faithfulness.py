"""Тесты модели-судьи: вердикт истинности разбирается из ответа модели."""
import asyncio

from app.schemas.cards import CardDraft, FaithfulnessVerdict
from app.services import pipeline


def _card() -> CardDraft:
    """Минимальная карточка для проверки судьёй."""
    return CardDraft(
        title="Чайник 2200 Вт",
        description="Чайник из нержавеющей стали",
        characteristics={"мощность": "2200 Вт"},
    )


def test_judge_verdict_parsed(monkeypatch):
    """Вердикт судьи из JSON модели: supported/unsupported на месте."""
    verdict = FaithfulnessVerdict(
        supported=["мощность 2200 Вт"], unsupported=["цвет белый"]
    )

    async def fake_run_agent(agent, prompt):
        return verdict.model_dump_json()

    monkeypatch.setattr(pipeline, "run_agent", fake_run_agent)
    result = asyncio.run(pipeline.judge_faithfulness("контекст", _card()))
    assert result.supported == ["мощность 2200 Вт"]
    assert result.unsupported == ["цвет белый"]


def test_judge_prompt_contains_context_and_card(monkeypatch):
    """Судья получает и контекст, и карточку в одном промпте."""
    prompts: list[str] = []

    async def fake_run_agent(agent, prompt):
        prompts.append(prompt)
        return FaithfulnessVerdict().model_dump_json()

    monkeypatch.setattr(pipeline, "run_agent", fake_run_agent)
    asyncio.run(pipeline.judge_faithfulness("фрагменты документов", _card()))
    assert "фрагменты документов" in prompts[0]
    assert "Чайник 2200 Вт" in prompts[0]


def test_judge_verdict_object_passthrough(monkeypatch):
    """SDK может вернуть готовый объект — он принимается без разбора."""
    verdict = FaithfulnessVerdict(supported=["всё"])

    async def fake_run_agent(agent, prompt):
        return verdict

    monkeypatch.setattr(pipeline, "run_agent", fake_run_agent)
    result = asyncio.run(pipeline.judge_faithfulness("контекст", _card()))
    assert result is verdict
