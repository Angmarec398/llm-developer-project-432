"""Пул соединений с Postgres.

Пул — ресурс уровня приложения: создаётся один раз на процесс
и живёт столько же, сколько процесс. Соединение — ресурс уровня
операции: берут из пула на короткую работу и возвращают.
"""
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

from app.core.config import settings

_pool: ConnectionPool | None = None


def init_pool() -> None:
    """Создаёт пул соединений. Вызывается один раз на старте процесса."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(settings.database_url, open=True, kwargs={"autocommit": True})


def close_pool() -> None:
    """Закрывает пул и возвращает соединения серверу. Вызывается при остановке."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def pool() -> ConnectionPool:
    """Возвращает пул соединений, создавая его при первом обращении."""
    if _pool is None:
        init_pool()
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Выдаёт соединение из пула контекстным менеджером.

    Соединение берётся на одну операцию и возвращается в пул
    по выходе из блока with — даже при исключении.
    """
    with pool().connection() as conn:
        yield conn
