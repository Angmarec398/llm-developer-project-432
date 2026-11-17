"""Repository-слой: таблица documents в Postgres.

Единственный владелец SQL для документов поставщиков.
Статусная машина: new → parsing → indexed | failed
(failed сопровождается причиной в поле error).
"""
import uuid
from datetime import UTC, datetime

import psycopg


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_document(
    conn: psycopg.Connection,
    filename: str,
    file_path: str,
    content_hash: str,
) -> tuple[str, bool]:
    """Создать документ с дедупликацией по хешу содержимого.

    Возвращает (document_id, created): created=True — документ создан
    сейчас, created=False — файл с таким содержимым уже загружался,
    вернулся id существующего документа (повторный разбор запускать
    нельзя — так повторная загрузка не создаёт дублей).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE content_hash = %s", (content_hash,)
        )
        row = cur.fetchone()
        if row:
            return row[0], False
        document_id = uuid.uuid4().hex[:12]
        cur.execute(
            """
            INSERT INTO documents (id, filename, file_path, content_hash)
            VALUES (%s, %s, %s, %s)
            """,
            (document_id, filename, file_path, content_hash),
        )
        return document_id, True


def get_document(conn: psycopg.Connection, document_id: str) -> dict | None:
    """Вернуть документ со счётчиком фрагментов (None, если документа нет)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.filename, d.status, d.error,
                   (SELECT count(*) FROM chunks c WHERE c.document_id = d.id)
                       AS chunks_count
            FROM documents d
            WHERE d.id = %s
            """,
            (document_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "status": row[2],
        "error": row[3],
        "chunks_count": row[4],
    }


def set_status(
    conn: psycopg.Connection,
    document_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """Переключить состояние документа; при отказе сохранить причину."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE documents
            SET status = %s, error = %s, updated_at = %s
            WHERE id = %s
            """,
            (status, error, _now(), document_id),
        )
