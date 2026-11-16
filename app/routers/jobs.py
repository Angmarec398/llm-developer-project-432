"""Фоновый вариант: job в RabbitMQ → Celery-воркер (шаги 2-3 практики)."""
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from app.core.db import connection
from app.schemas.cards import JobIn
from app.repositories import jobs as jobs_repo
from app.workers.celery_app import celery_app

router = APIRouter(tags=["jobs"])


@router.post("/jobs", status_code=202)
async def create_job(
    body: JobIn,
    idempotency_key: str | None = Header(
        default=None, alias="Idempotency-Key"),
):
    """Создание задачи в очереди"""
    with connection() as conn:
        try:
            job_id = jobs_repo.create_job(
                conn, {"supplier_text": body.supplier_text}, idempotency_key
            )
        finally:
            conn.close()
        celery_app.send_task(
            "app.workers.tasks.generate_card",
            kwargs={"job_id": job_id, "supplier_text": body.supplier_text},
            queue="card.generate",
        )
        return JSONResponse({"id": job_id, "status": "pending"}, status_code=202)



@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    with connection() as conn:
        try:
            job = jobs_repo.get_job(conn, job_id)
        finally:
            conn.close()
        if not job:
            raise HTTPException(404, "job not found")
        return job
    
