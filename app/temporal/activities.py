"""Temporal-activities: те же стадии пайплайна из app/services.

Activities выполняются БЕЗ песочницы — здесь можно импортировать конфиг,
LLM-клиент, ходить в сеть. Данные передаём JSON-строками: Temporal
сериализует их в историю workflow, Pydantic-валидация — на границе.
"""
import json

from temporalio import activity

from app.core.db import connection
from app.repositories import jobs as jobs_repo
from app.schemas.cards import ProductCard, SupplierFacts
from app.services import pipeline
from app.services.documents import process_document


@activity.defn
def set_status_activity(
    job_id: str,
    status: str,
    result_json: str | None = None,
    error: str | None = None,
    bump_attempts: bool = False,
) -> None:
    """Записать статус задачи в БД через репозиторий (отдельный шаг процесса)."""
    result = ProductCard.model_validate_json(result_json) if result_json else None
    with connection() as conn:
        jobs_repo.set_status(
            conn, job_id, status,
            result=result, error=error, bump_attempts=bump_attempts,
        )


@activity.defn
async def generate_activity(facts_json: str, feedback_json: str | None = None) -> str:
    """Сгенерировать карточку по фактам (шаг generate, контракт шага 4)."""
    facts = SupplierFacts.model_validate_json(facts_json)
    feedback = json.loads(feedback_json) if feedback_json else None
    card = await pipeline.generate_card(facts, feedback)
    return card.model_dump_json()


@activity.defn
async def critique_activity(facts_json: str, draft_json: str) -> str:
    """Проверить черновик по правилам и вернуть вердикт (шаг critique)."""
    facts = SupplierFacts.model_validate_json(facts_json)
    draft = ProductCard.model_validate_json(draft_json)
    report = await pipeline.critique(facts, draft)
    return report.model_dump_json()


@activity.defn
async def extract_activity(supplier_text: str) -> str:
    """Извлечь структурированные факты из текста поставщика (шаг extract)."""
    facts = await pipeline.extract(supplier_text)
    return facts.model_dump_json()


@activity.defn
def parse_document_activity(document_id: str, file_path: str) -> int:
    """Разобрать документ в фрагменты (шаг процесса DocumentWorkflow).

    Синхронная и блокирующая (диск + база): выполняется в потоке
    пула activity_executor, как и любой шаг с базой или диском.
    Ошибки разбора не летят наружу — сервис превращает их в отказ
    документа с причиной и возвращает 0.
    """
    return process_document(document_id, file_path)
