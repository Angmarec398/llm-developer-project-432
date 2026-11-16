from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.db import close_pool, init_pool
from app.core.logging import setup_logging
from app.llm.client import setup_llm
from app.routers import cards, health, jobs


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
app.include_router(health.router)