"""Раннер миграций: применяет SQL-файлы из db/migrations по порядку.

Журнал применённых версий — таблица schema_migrations.
Каждая миграция выполняется в своей транзакции: упавшая не попадает
в журнал и применится при следующем запуске.
"""
from pathlib import Path

import psycopg

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "db" / "migrations"

_journal_ddl = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _applied_versions(conn: psycopg.Connection) -> set[str]:
    """Возвращает версии миграций, уже записанные в журнал.

    Перед чтением создаёт таблицу schema_migrations, если её ещё нет:
    на первом запуске журнал отсутствует, а опрашивать его нужно
    до применения самой первой миграции.
    """
    with conn.cursor() as cur:
        cur.execute(_journal_ddl)
        cur.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def _apply(conn: psycopg.Connection, path: Path) -> None:
    """Применяет одну миграцию и отмечает её версию в журнале.

    Файл выполняется целиком в одной транзакции вместе с записью
    в schema_migrations: при ошибке оба изменения откатываются,
    и миграция применится повторно при следующем запуске.
    """
    version = path.stem
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(path.read_text(encoding="utf-8"))
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
        )
    print(f"применена миграция: {version}")
    log.info("migration_applied", version=version)


def main() -> None:
    """Точка входа: применяет все ещё не применённые миграции по порядку имён.

    Печатает имя каждой применённой миграции; если новых нет,
    сообщает об этом и завершается без изменений.
    """
    with psycopg.connect(settings.database_url) as conn:
        applied = _applied_versions(conn)
        pending = [
            path
            for path in sorted(MIGRATIONS_DIR.glob("*.sql"))
            if path.stem not in applied
        ]
        for path in pending:
            _apply(conn, path)
    if not pending:
        print("новых миграций нет")


if __name__ == "__main__":
    main()
