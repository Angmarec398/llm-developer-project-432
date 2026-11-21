"""Репозиторий учёта вызовов модели (таблица llm_calls, шаг 8).

SQL живёт только здесь. Клиент модели пишет через record_call после
каждого вызова; cost_by_model — разбивка для отчёта: сколько вызовов
и на какую сумму ушло на каждую модель.
"""
from datetime import datetime
from decimal import Decimal

from app.core.db import connection


def record_call(
    job_id: str | None,
    agent: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: Decimal,
    duration_ms: int,
    request_id: str | None = None,
    created_at: datetime | None = None,
) -> None:
    """Записать один вызов модели в таблицу учёта.

    job_id из контекста задачи (None — вне задачи, напр. ручной прогон
    метрик); request_id — сквозной идентификатор HTTP-запроса;
    created_at пробрасывается из тестов, в продакшене — now() из базы
    (COALESCE: переданный NULL не отменяет DEFAULT).
    """
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO llm_calls
                (job_id, request_id, agent, model,
                 prompt_tokens, completion_tokens, cost, duration_ms,
                 created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()))
            """,
            (
                job_id,
                request_id,
                agent,
                model,
                prompt_tokens,
                completion_tokens,
                cost,
                duration_ms,
                created_at,
            ),
        )


def cost_by_model(conn) -> list[dict]:
    """Разбивка по моделям: вызовы, суммарные токены, суммарная стоимость.

    Один SQL-запрос — и видно экономию двухмодельной политики:
    сколько ушло на дешёвую служебную, сколько на основную.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT model,
                   COUNT(*)            AS calls,
                   SUM(prompt_tokens)  AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(cost)           AS total_cost
            FROM llm_calls
            GROUP BY model
            ORDER BY total_cost DESC
            """
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
