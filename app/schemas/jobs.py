from pydantic import BaseModel

from app.schemas.cards import CardDraft, GuardrailReport


class JobOut(BaseModel):
    """Ответ GET /jobs/{id}: статус задачи, итоговая карточка и отчёт
    защиты: что замаскировано, что исключено, что в карантине."""

    id: str
    status: str
    attempts: int
    result: CardDraft | None = None
    error: str | None = None
    guardrail_report: GuardrailReport | None = None
