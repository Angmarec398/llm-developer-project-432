"""Стадии пайплайна: extract → generate → critique.

Каждая функция — один агент-шаг. Один и тот же код работает в трёх runtime:
- FastAPI-эндпоинт (шаг 1 практики),
- Celery-воркер (шаги 2-3),
- Temporal activity (шаг 4).
Это и есть выигрыш слоистой архитектуры.
"""
from app.agents.prompts import card_agent, critic_agent, extractor_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.cards import CritiqueReport, ProductCard, SupplierFacts
from app.services.structured import call_validated

log = get_logger(__name__)


async def extract(supplier_text: str) -> SupplierFacts:
    """Извлечь факты из текста поставщика (с лечением ответа)."""
    log.info("step_extract_start")
    return await call_validated(extractor_agent, supplier_text, SupplierFacts)


async def generate_card(
    facts: SupplierFacts,
    feedback: list[str] | None = None,
    source_name: str = "supplier_text",
) -> ProductCard:
    """Сгенерировать карточку по контракту шага 4 (с лечением ответа)."""
    prompt = (
        "Факты о товаре:\n"
        + facts.model_dump_json(indent=2)
        + f"\nИсточник: {source_name}"
    )
    if feedback:
        prompt += "\n\nЗамечания критика, учти:\n" + "\n".join(
            f"- {issue}" for issue in feedback
        )
    log.info("step_generate_start", feedback=bool(feedback))
    return await call_validated(card_agent, prompt, ProductCard)


async def critique(facts: SupplierFacts, card: ProductCard) -> CritiqueReport:
    """Проверить карточку по правилам (с лечением ответа)."""
    prompt = (
        "Факты о товаре:\n"
        + facts.model_dump_json(indent=2)
        + "\n\nКарточка для проверки:\n"
        + card.model_dump_json(indent=2)
    )
    log.info("step_critique_start")
    return await call_validated(critic_agent, prompt, CritiqueReport)


async def run_pipeline(supplier_text: str, max_attempts: int = 3):
    """Полный цикл: extract → (generate → critique → feedback)* → итог.

    Возвращает (карточку, число генераций, вердикт).
    Вердикт: approved — критик доволен и уверенность выше порога;
    low_confidence — критик доволен, но уверенность ниже порога,
    карточку должен посмотреть человек; rejected — петля исчерпана.
    """
    facts = await extract(supplier_text)
    feedback: list[str] = []
    attempts = 0
    card = None
    verdict = "rejected"
    while attempts < max_attempts:
        attempts += 1
        card = await generate_card(facts, feedback or None)
        report = await critique(facts, card)
        if report.verdict == "approve":
            verdict = "approved"
            break
        feedback = report.issues
        log.info("pipeline_regenerate", attempt=attempts, issues=report.issues)
    if (
        verdict == "approved"
        and card.confidence < settings.card_confidence_threshold
    ):
        verdict = "low_confidence"
        log.info("pipeline_low_confidence", confidence=card.confidence)
    return card, attempts, verdict
