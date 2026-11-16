from pydantic import BaseModel

from app.schemas.cards import CardDraft


class JobOut(BaseModel):
    id: str
    status: str
    attempts: int
    result: CardDraft | None = None
    error: str | None = None
