from contextlib import asynccontextmanager
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request

from app.core.context import bind_request
from app.core.db import close_pool, init_pool
from app.core.logging import setup_logging
from app.llm.client import setup_llm
from app.routers import cards, documents, health, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_pool()
    setup_llm()
    yield
    close_pool()


app = FastAPI(title="card-pipeline", lifespan=lifespan)
app.include_router(cards.router)
app.include_router(jobs.router)
app.include_router(documents.router)
app.include_router(health.router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Сквозной идентификатор запроса (шаг 8): рождается на входе в приложение.

    Принимаем X-Request-ID от клиента или генерируем свой; дальше id живёт
    в contextvars — попадает в каждую строку логов запроса и в таблицу
    вызовов модели. Возвращаем заголовком, чтобы клиент мог сослаться
    на свой запрос при разборе инцидента.
    """
    request_id = request.headers.get("x-request-id") or uuid4().hex
    with bind_request(request_id):
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
