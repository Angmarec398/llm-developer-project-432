"""Синхронный вариант: весь сквозной сценарий внутри запроса.

Тот же код оркестрации, что и в устойчивом процессе
(app/services/cards.py), только без Temporal и статусов в БД —
для ручной проверки и отладки.
"""
from fastapi import APIRouter, HTTPException

from app.schemas.cards import JobIn, ProductCard
from app.services import cards as cards_service

router = APIRouter(tags=["cards"])


@router.post("/cards", response_model=ProductCard)
async def create_card(body: JobIn) -> ProductCard:
    """Сгенерировать карточку синхронно: документы → поиск → генерация.

    Проблемы с документами (не найдены, разобраны с ошибкой) — 422
    с причиной, а не падение с 500. Отчёт защиты (шаг 9) в ответе
    эндпоинта не нужен: возвращается только карточка.
    """
    try:
        card, _attempts, _verdict, _report = (
            await cards_service.run_card_pipeline(body.document_ids, body.hint)
        )
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return card
