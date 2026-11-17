from pydantic import BaseModel

from app.schemas.cards import ProductCard


class JobOut(BaseModel):
    """Ответ GET /jobs/{id}: статус задачи и итоговая карточка."""

    id: str
    status: str
    attempts: int
    result: ProductCard | None = None
    error: str | None = None
