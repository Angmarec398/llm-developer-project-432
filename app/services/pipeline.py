"""Стадии пайплайна шага 7: generate → critique, генерация от контекста.

Контекст — найденные фрагменты документов (app/rag), а не целый текст
поставщика: extractor-стадия больше не нужна, её работу делает поиск.
Модель обязана вернуть источники (фрагменты с id); проверка ссылок —
детерминированный код (app/services/citations.py), критик сверяет
утверждения с источниками.

Стадии мелкие и самостоятельные — цикл перегенерации живёт в
оркестрации (app/services/cards.py), чтобы один и тот же код работал
в двух runtime: FastAPI-эндпоинте (POST /cards) и Temporal activity.
"""
from app.agents.prompts import card_agent, critic_agent
from app.core.logging import get_logger
from app.schemas.cards import CritiqueReport, ProductCard
from app.services.structured import call_validated

log = get_logger(__name__)


async def generate_card(
    context: str, feedback: list[str] | None = None
) -> ProductCard:
    """Сгенерировать карточку от контекста фрагментов (с лечением ответа).

    feedback — замечания к переделке (нарушения ссылок или список
    проблем критика): уходят в промпт, чтобы модель исправила именно
    их, а не переписала карточку с нуля.
    """
    prompt = "Контекст — фрагменты документов поставщика:\n" + context
    if feedback:
        prompt += "\n\nЗамечания к переделке, учти каждое:\n" + "\n".join(
            f"- {issue}" for issue in feedback
        )
    log.info("step_generate_start", feedback=bool(feedback))
    return await call_validated(card_agent, prompt, ProductCard)


async def critique(context: str, card: ProductCard) -> CritiqueReport:
    """Проверить карточку: критик сверяет утверждения с источниками."""
    prompt = (
        "Контекст — фрагменты документов поставщика:\n"
        + context
        + "\n\nКарточка для проверки:\n"
        + card.model_dump_json(indent=2)
    )
    log.info("step_critique_start")
    return await call_validated(critic_agent, prompt, CritiqueReport)
