"""Сквозные идентификаторы запроса и задачи (шаг 8).

request_id рождается на входе в приложение (middleware), job_id — при
создании задачи и уходит в процесс аргументом. Оба живут в contextvars:
видны любому коду внутри запроса или шага процесса — логам и клиенту
модели. Смысл: один поиск по идентификатору задачи показывает весь
её путь — обработчик, шаги процесса, вызовы модели с длительностью.
"""
import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "job_id", default=None
)


def current_request_id() -> str | None:
    """Идентификатор HTTP-запроса, в рамках которого выполняется код."""
    return request_id_var.get()


def current_job_id() -> str | None:
    """Идентификатор задачи (jobs.id), в рамках которой выполняется код."""
    return job_id_var.get()


@contextmanager
def bind_request(request_id: str) -> Iterator[None]:
    """Привязать request_id к контексту на время блока (middleware)."""
    reset_token = request_id_var.set(request_id)
    try:
        yield
    finally:
        request_id_var.reset(reset_token)


@contextmanager
def bind_job(job_id: str) -> Iterator[None]:
    """Привязать job_id к контексту на время блока (воркер, эндпоинт)."""
    reset_token = job_id_var.set(job_id)
    try:
        yield
    finally:
        job_id_var.reset(reset_token)
