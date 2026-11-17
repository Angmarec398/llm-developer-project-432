"""Синхронный вариант: весь пайплайн внутри запроса (шаг 1 практики)."""
from fastapi import APIRouter

from app.schemas.cards import JobIn, ProductCard
from app.services.pipeline import run_pipeline

router = APIRouter(tags=["cards"])


@router.post("/cards", response_model=ProductCard)
async def create_card(body: JobIn) -> ProductCard:
    """Сгенерировать карточку синхронно, внутри HTTP-запроса."""
    card, _attempts, _verdict = await run_pipeline(body.supplier_text)
    return card
