"""Repository-слой: таблицы documents и chunks в Postgres.

Единственный владелец SQL для документов поставщиков и их фрагментов.
Статусная машина: new → parsing → indexed | failed
(failed сопровождается причиной в поле error).

get_document отдаёт ещё и file_path — оркестрации карточки нужен путь,
чтобы доразобрать документ, загруженный, но ещё не проиндексированный.
"""
import json
import uuid
from datetime import UTC, datetime

import psycopg


def _now() -> str:
    """Текущее время в ISO — для полей updated_at."""
    return datetime.now(UTC).isoformat()


def create_document(
    conn: psycopg.Connection,
    filename: str,
    file_path: str,
    content_hash: str,
    kind: str = "",
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
            INSERT INTO documents (id, filename, file_path, content_hash, kind)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (document_id, filename, file_path, content_hash, kind),
        )
        return document_id, True


def get_document(conn: psycopg.Connection, document_id: str) -> dict | None:
    """Вернуть документ с путём, статусом и счётчиком фрагментов.

    None, если документа нет. file_path нужен оркестрации карточки
    для доразбора непроиндексированного документа.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, file_path, kind, status, chunks_count, error, created_at
            FROM documents
            WHERE id = %s
            """,
            (document_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "file_path": row[2],
        "kind": row[3],
        "status": row[4],
        "chunks_count": row[5],
        "error": row[6],
        "created_at": str(row[7]),
    }


def list_documents(conn: psycopg.Connection) -> list[dict]:
    """Список всех документов по времени создания."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, kind, status, chunks_count FROM documents ORDER BY created_at"
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "filename": r[1], "kind": r[2], "status": r[3], "chunks_count": r[4]}
        for r in rows
    ]


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


def set_chunks_count(conn: psycopg.Connection, doc_id: str, count: int) -> None:
    """Зафиксировать число фрагментов документа после разбора."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET chunks_count = %s, updated_at = %s WHERE id = %s",
            (count, _now(), doc_id),
        )


def insert_chunks(conn: psycopg.Connection, chunks: list[dict]) -> None:
    """Батч-вставка чанков. Транзакция: либо все, либо ничего."""
    if not chunks:
        return
    rows = [
        (
            chunk["id"],
            chunk["metadata"]["doc_id"],
            chunk["ordinal"],
            chunk["text"],
            json.dumps(chunk["metadata"], ensure_ascii=False),
        )
        for chunk in chunks
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, doc_id, ordinal, text, metadata) VALUES (%s, %s, %s, %s, %s)",
            rows,
        )


def delete_chunks(conn: psycopg.Connection, doc_id: str) -> None:
    """Удалить все фрагменты документа (перед пересозданием при повторном разборе)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
