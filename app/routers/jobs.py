"""Фоновый вариант: задача в БД + устойчивый процесс в Temporal.

Идемпотентность двойная: на уровне БД (ключ идемпотентности) и на
уровне движка (workflow_id детерминированно выводится из job_id —
повторный старт с тем же id Temporal не выполнит).

Шаг 7: задача принимает список документов и необязательную подсказку
о товаре; результат читается вместе со ссылками на источники
(GET /jobs/{id}).
"""
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from temporalio.service import RPCError, RPCStatusCode

from app.core.config import settings
from app.core.db import connection
from app.repositories import jobs as jobs_repo
from app.schemas.cards import JobIn
from app.temporal.client import get_client

router = APIRouter(tags=["jobs"])


async def _signal(job_id: str, signal: str) -> None:
    """Послать сигнал процессу задачи.

    404 — процесса с таким id нет, 409 — процесс уже завершён.
    """
    client = await get_client()
    handle = client.get_workflow_handle(f"card-{job_id}")
    try:
        await handle.signal(signal)
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            if "already completed" in str(exc):
                raise HTTPException(409, "workflow already completed") from exc
            raise HTTPException(404, "workflow not found") from exc
        raise


@router.post("/jobs", status_code=202)
async def create_job(
    body: JobIn,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """Создать задачу генерации карточки и запустить её процесс в движке.

    Принимает выбранные документы и подсказку о товаре, отвечает 202
    и идентификатором задачи; результат (карточка со ссылками на
    фрагменты-источники) читается через GET /jobs/{id}. Повтор с тем
    же Idempotency-Key возвращает существующую задачу без нового
    запуска процесса.
    """
    payload = {"document_ids": body.document_ids, "hint": body.hint}
    with connection() as conn:
        job_id, created = jobs_repo.create_job(conn, payload, idempotency_key)
    if created:
        client = await get_client()
        await client.start_workflow(
            "CardWorkflow",
            args=[
                job_id,
                body.document_ids,
                body.hint,
                3,
                settings.card_confidence_threshold,
            ],
            id=f"card-{job_id}",
            task_queue=settings.temporal_task_queue,
        )
    return JSONResponse({"id": job_id, "status": "pending"}, status_code=202)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Вернуть статус задачи и карточку со ссылками на источники."""
    with connection() as conn:
        job = jobs_repo.get_job(conn, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@router.post("/jobs/{job_id}/approve")
async def approve_job(job_id: str):
    """Подтвердить карточку: процесс завершится со статусом approved.

    Работает и для карточки, эскалированной за повторные выдуманные
    ссылки, — решение остаётся за человеком.
    """
    await _signal(job_id, "approve")
    return {"id": job_id, "signaled": "approve"}


@router.post("/jobs/{job_id}/reject")
async def reject_job(job_id: str):
    """Отклонить карточку: процесс завершится со статусом rejected."""
    await _signal(job_id, "request_changes")
    return {"id": job_id, "signaled": "request_changes"}
