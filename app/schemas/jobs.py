from pydantic import BaseModel

from app.schemas.cards import GuardrailReport, ProductCard


class JobOut(BaseModel):
    """Ответ GET /jobs/{id}: статус задачи, итоговая карточка и отчёт
    защиты (шаг 9): что замаскировано, что исключено, что в карантине."""

    id: str
    status: str
    attempts: int
    result: ProductCard | None = None
    error: str | None = None
    guardrail_report: GuardrailReport | None = None
