"""Repository-слой: таблица chunks в Postgres.

Фрагменты документа вставляются одним пакетом после разбора;
seq — порядок фрагмента в документе, назначается здесь по позиции
в списке, чтобы вызывающий код о нём не думал.
"""
import psycopg


def save_chunks(
    conn: psycopg.Connection,
    document_id: str,
    chunks: list[dict],
) -> int:
    """Массово вставить фрагменты документа, вернуть число вставленных.

    Каждый элемент chunks — dict с ключами text, page, section,
    article, brand (метаданные могут быть None).
    """
    rows = [
        (
            document_id,
            position,
            chunk["text"],
            chunk.get("page"),
            chunk.get("section"),
            chunk.get("article"),
            chunk.get("brand"),
        )
        for position, chunk in enumerate(chunks)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (document_id, seq, text, page, section, article, brand)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    return len(rows)
