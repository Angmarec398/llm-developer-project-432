"""Repository-слой: таблица jobs в Postgres.

Единственный владелец SQL. Статусная машина:
pending → extracting → generating → critiquing → (regenerating)
→ done | awaiting_confirmation → approved | rejected | failed
"""
import json
import uuid
from datetime import UTC, datetime

import psycopg
from pydantic import ValidationError

from app.schemas.cards import CardDraft, GuardrailReport


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_job(
    conn: psycopg.Connection, payload: dict, idempotency_key: str | None
) -> tuple[str, bool]:
    """Создать задачу с ключом идемпотентности.

    Возвращает (job_id, created): created=True — задача создана сейчас,
    created=False — задача с таким ключом уже существовала, вернулся её id
    (повторный запуск фоновой работы в этом случае делать нельзя).
    """
    with conn.cursor() as cur:
        if idempotency_key:
            cur.execute(
                "SELECT id FROM jobs WHERE idempotency_key = %s", (idempotency_key,)
            )
            row = cur.fetchone()
            if row:
                return row[0], False
        job_id = uuid.uuid4().hex[:12]
        cur.execute(
            "INSERT INTO jobs (id, idempotency_key, payload) VALUES (%s, %s, %s)",
            (job_id, idempotency_key, json.dumps(payload)),
        )
        return job_id, True



def set_status(
    conn: psycopg.Connection, job_id: str, status: str, *, result: CardDraft | None = None,
    error: str | None = None, bump_attempts: bool = False,
    guardrail_report: GuardrailReport | None = None,
) -> None:
    """Сохранение информации о статусе обработки заявки в базе данных"""
    status_info = ["status = %s", "updated_at = %s"]
    args: list = [status, _now()]
    if result is not None:
        status_info.append("result = %s")
        args.append(result.model_dump_json())
    if guardrail_report is not None:
        status_info.append("guardrail_report = %s")
        args.append(guardrail_report.model_dump_json())
    if error is not None:
        status_info.append("error = %s")
        args.append(error)
    if bump_attempts:
        status_info.append("attempts = attempts + 1")
    args.append(job_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE jobs SET {', '.join(status_info)} WHERE id = %s", args)



def get_job(conn: psycopg.Connection, job_id: str) -> dict | None:
    """Возвращает информацию по задаче на основании id"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, status, attempts, result, error, guardrail_report "
            "FROM jobs WHERE id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    job_info = {"id": row[0], "status": row[1], "attempts": row[2], "error": row[4]}
    if row[3]:
        # psycopg3 возвращает JSONB уже как dict, а не строку
        try:
            draft = (
                CardDraft.model_validate(row[3])
                if isinstance(row[3], dict)
                else CardDraft.model_validate_json(row[3])
            )
            job_info["result"] = draft.model_dump()
        except ValidationError:
            job_info["result"] = None
    if row[5]:
        try:
            report = (
                GuardrailReport.model_validate(row[5])
                if isinstance(row[5], dict)
                else GuardrailReport.model_validate_json(row[5])
            )
            job_info["guardrail_report"] = report.model_dump()
        except ValidationError:
            job_info["guardrail_report"] = None
    return job_info

