"""Проверки живости и готовности приложения."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.db import connection

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness():
    """Живость: процесс отвечает. Базы не касается — успех даже при остановленной БД."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness():
    """Готовность: база доступна и расширение vector активно.

    Соединение берётся из пула на одну короткую операцию.
    """
    try:
        with connection() as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
    except Exception:
        return JSONResponse(
            {"status": "error", "database": "unavailable"}, status_code=503
        )
    if not row:
        return JSONResponse({"status": "error", "vector": "inactive"}, status_code=503)
    return {"status": "ok", "database": "up", "vector": row[0]}
