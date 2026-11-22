"""Temporal-activities: шаги сквозного сценария из app/services.

Activities выполняются БЕЗ песочницы — здесь можно импортировать конфиг,
LLM-клиент, ходить в сеть и базу. Данные передаём JSON-строками: Temporal
сериализует их в историю workflow, Pydantic-валидация — на границе.

Каждая activity — ровно один вызов функции оркестрации
(app/services/cards.py): так сквозной сценарий одинаков в синхронном
эндпоинте POST /cards и в устойчивом процессе CardWorkflow, а шаги
процесса видны в его истории.

Учёт и трассировка (шаг 8): LLM-activities принимают job_id первым
аргументом и привязывают его к contextvars — вызовы модели внутри
попадают в таблицу llm_calls с этим job_id, а логи шага можно найти
одним поиском.
"""
import json

import structlog
from temporalio import activity

from app.core.context import bind_job
from app.core.db import connection
from app.guardrails.output import filter_card_output
from app.repositories import jobs as jobs_repo
from app.schemas.cards import GuardrailReport, ProductCard
from app.services import cards as cards_service
from app.services import pipeline
from app.services.documents import process_document


@activity.defn
def set_status_activity(
    job_id: str,
    status: str,
    result_json: str | None = None,
    error: str | None = None,
    bump_attempts: bool = False,
    report_json: str | None = None,
) -> None:
    """Записать статус задачи в БД через репозиторий (отдельный шаг процесса).

    report_json — отчёт защиты (шаг 9): пишется вместе с финальными
    статусами, оператор видит, что замаскировано и что исключено.
    """
    result = ProductCard.model_validate_json(
        result_json) if result_json else None
    report = (
        GuardrailReport.model_validate_json(
            report_json) if report_json else None
    )
    with connection() as conn:
        jobs_repo.set_status(
            conn, job_id, status,
            result=result, error=error, bump_attempts=bump_attempts,
            guardrail_report=report,
        )


@activity.defn
def parse_documents_activity(document_ids_json: str) -> str:
    """Разобрать непроиндексированные документы в фрагменты (шаг parsing).

    Возвращает JSON {"problems": [...], "parsed": {doc_id: число фрагментов}}.
    Непустой problems — процесс ставит задаче failed с этими причинами:
    документ не найден или разобран с ошибкой, генерировать не из чего.
    """
    document_ids = json.loads(document_ids_json)
    problems = cards_service.documents_problems(document_ids)
    parsed = (
        {} if problems
        else cards_service.parse_pending_documents(document_ids)
    )
    return json.dumps(
        {"problems": problems, "parsed": parsed}, ensure_ascii=False
    )


@activity.defn
def embed_documents_activity(document_ids_json: str) -> int:
    """Досчитать эмбеддинги по выбранным документам (шаг indexing).

    Идемпотентно: вектор ставится только тем фрагментам, у которых
    его нет. Возвращает число записанных векторов.
    """
    return cards_service.embed_pending_documents(json.loads(document_ids_json))


@activity.defn
def search_chunks_activity(document_ids_json: str, hint: str | None) -> str:
    """Найти фрагменты и замаскировать ПДн (шаг searching, шаг 9).

    Маскирование — до того, как фрагменты покидают activity: в историю
    процесса Temporal попадает только замаскированный текст. Возвращает
    JSON {"chunks": [...], "report": {...}}; отчёт — только метки,
    без сырых значений.
    """
    chunks, report = cards_service.search_chunks(
        json.loads(document_ids_json), hint
    )
    return json.dumps(
        {"chunks": chunks, "report": report.model_dump()}, ensure_ascii=False
    )


@activity.defn
async def screen_context_activity(job_id: str, found_json: str) -> str:
    """Скрининг фрагментов детектором инъекций (шаг screening, шаг 9).

    Подозрительные фрагменты выбрасываются из контекста; документ с
    превышением порога подозрительных — целиком к человеку, тогда
    blocked_reason не пуст, а контекст пустой. Возвращает JSON
    {"context", "chunk_ids", "report", "blocked_reason"}. job_id
    привязывается к контексту на время шага: вызовы служебной модели
    детектора уходят в таблицу учёта с этим job_id, логи шага
    находятся одним поиском.
    """
    found = json.loads(found_json)
    report = GuardrailReport.model_validate(found["report"])
    with bind_job(job_id):
        structlog.contextvars.bind_contextvars(job_id=job_id)
        screened, report, blocked_reason = await cards_service.screen_chunks(
            found["chunks"], report
        )
    context, chunk_ids = cards_service.assemble_context(screened)
    return json.dumps(
        {
            "context": context,
            "chunk_ids": chunk_ids,
            "report": report.model_dump(),
            "blocked_reason": blocked_reason,
        },
        ensure_ascii=False,
    )


@activity.defn
async def generate_activity(
    job_id: str, context: str, feedback_json: str | None = None
) -> str:
    """Сгенерировать карточку от контекста фрагментов (шаг generating).

    job_id привязывается к контексту на время шага: вызовы модели внутри
    уходят в таблицу учёта с этим job_id, логи шага находятся поиском.
    """
    feedback = json.loads(feedback_json) if feedback_json else None
    with bind_job(job_id):
        structlog.contextvars.bind_contextvars(job_id=job_id)
        card = await pipeline.generate_card(context, feedback)
    return card.model_dump_json()


@activity.defn
def verify_citations_activity(draft_json: str, context_chunk_ids_json: str) -> str:
    """Проверить ссылки карточки (шаг verifying): JSON-список нарушений.

    Каждая ссылка проверяется по двум условиям: фрагмент существует
    в базе и был в поданном контексте. Пустой список — ссылки в порядке.
    """
    card = ProductCard.model_validate_json(draft_json)
    violations = cards_service.verify_card_citations(
        card, json.loads(context_chunk_ids_json)
    )
    return json.dumps(violations, ensure_ascii=False)


@activity.defn
async def critique_activity(job_id: str, context: str, draft_json: str) -> str:
    """Проверить черновик критиком (шаг critiquing): вердикт и проблемы.

    job_id привязывается к контексту на время шага (учёт и логи).
    """
    draft = ProductCard.model_validate_json(draft_json)
    with bind_job(job_id):
        structlog.contextvars.bind_contextvars(job_id=job_id)
        report = await pipeline.critique(context, draft)
    return report.model_dump_json()


@activity.defn
def parse_document_activity(document_id: str, file_path: str) -> int:
    """Разобрать документ в фрагменты (шаг процесса DocumentWorkflow).

    Синхронная и блокирующая (диск + база): выполняется в потоке
    пула activity_executor, как и любой шаг с базой или диском.
    Ошибки разбора не летят наружу — сервис превращает их в отказ
    документа с причиной и возвращает 0.
    """
    return process_document(document_id, file_path)


@activity.defn
def filter_output_activity(draft_json: str, report_json: str) -> str:
    """Фильтр выхода (шаг 9): маскировать ПДн в карточке, найти следы инъекций.

    Последний рубеж перед отдачей: карточка прогоняется теми же
    детекторами, что фрагменты на входе. Возвращает JSON
    {"card": <очищенный черновик>, "report": <отчёт>, "injection_traces":
    [...]}; непустые traces — процесс уходит к человеку с причиной.
    """
    card = ProductCard.model_validate_json(draft_json)
    report = GuardrailReport.model_validate_json(report_json)
    cleaned, traces = filter_card_output(card, report)
    return json.dumps(
        {
            "card": cleaned.model_dump(),
            "report": report.model_dump(),
            "injection_traces": traces,
        },
        ensure_ascii=False,
    )
