"""Repository: векторный upsert и три режима поиска (vector / fts / hybrid RRF).

Единственный владелец SQL для поиска по фрагментам. embed_query импортирован
на уровень модуля — тесты подменяют его фиксированными векторами, модель
при этом не грузится.
"""
import psycopg

from app.core.config import settings
from app.rag.embedder import embed_query

# rrf_k из классической формулы Reciprocal Rank Fusion
RRF_K = 60


def upsert_embeddings(conn: psycopg.Connection, items: list[tuple[str, list[float]]]) -> int:
    """Батч-запись эмбеддингов: [(chunk_id, vector)]. Возвращает число записанных."""
    if not items:
        return 0
    # vector передаём строкой '[1,2,...]' — psycopg не знает тип vector
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
            [(_vec_str(vector), chunk_id) for chunk_id, vector in items],
        )
    return len(items)


def _vec_str(vector: list[float]) -> str:
    """Вектор → строка '[x,y,...]' для каста ::vector в SQL."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def chunks_without_embedding(
    conn: psycopg.Connection, limit: int = 256, doc_id: str | None = None
) -> list[tuple[str, str]]:
    """Чанки без вектора: (id, text). Для досчёта эмбеддингов (reindex/ingestion)."""
    sql = "SELECT id, text FROM chunks WHERE embedding IS NULL"
    args: list = []
    if doc_id:
        sql += " AND doc_id = %s"
        args.append(doc_id)
    sql += " LIMIT %s"
    args.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def search_vector(
    conn: psycopg.Connection, query: str, *, top_k: int | None = None,
    doc_ids: list[str] | None = None, threshold: float | None = None,
) -> list[dict]:
    """Top-K по косинусной дистанции (<=>); порог фильтрует после отбора с запасом."""
    top_k = top_k or settings.retrieval_top_k
    threshold = settings.retrieval_threshold if threshold is None else threshold
    vec = _vec_str(embed_query(query))
    sql = """
        SELECT c.id, c.doc_id, c.text, c.metadata,
               d.filename,
               1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE c.embedding IS NOT NULL AND d.status = 'indexed'
    """
    args: list = [vec]
    if doc_ids:
        sql += " AND c.doc_id = ANY(%s)"
        args.append(doc_ids)
    sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
    args += [vec, top_k * 3]  # берём с запасом до фильтра по threshold
    with conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    return [
        _row_to_dict(row) for row in rows
        if row[5] is not None and row[5] >= threshold
    ][:top_k]


def search_fts(
    conn: psycopg.Connection, query: str, *, top_k: int | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """Полнотекстовый поиск по tsvector ('russian')."""
    top_k = top_k or settings.retrieval_top_k
    sql = """
        SELECT c.id, c.doc_id, c.text, c.metadata,
               d.filename,
               ts_rank(c.tsv, q.query) AS score
        FROM chunks c JOIN documents d ON d.id = c.doc_id,
             plainto_tsquery('russian', %s) AS q(query)
        WHERE c.tsv @@ q.query AND d.status = 'indexed'
    """
    args: list = [query]
    if doc_ids:
        sql += " AND c.doc_id = ANY(%s)"
        args.append(doc_ids)
    sql += " ORDER BY score DESC LIMIT %s"
    args.append(top_k)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def search_hybrid(
    conn: psycopg.Connection, query: str, *, top_k: int | None = None,
    doc_ids: list[str] | None = None, threshold: float | None = None,
) -> list[dict]:
    """Гибрид: векторный + полнотекстовый, слияние рангов через RRF.

    score = sum(1 / (rrf_k + rank_i)) по спискам результатов.
    CTE: каждый поиск — свой SELECT с ROW_NUMBER(), JOIN по chunk_id.
    """
    top_k = top_k or settings.retrieval_top_k
    vec = _vec_str(embed_query(query))
    sql = """
        WITH vec AS (
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS rank
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.embedding IS NOT NULL AND d.status = 'indexed'
    """
    args: list = [vec]
    if doc_ids:
        sql += " AND c.doc_id = ANY(%s)"
        args.append(doc_ids)
    sql += """
            LIMIT %s
        ), fts AS (
            SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank(c.tsv, q.query) DESC) AS rank
            FROM chunks c JOIN documents d ON d.id = c.doc_id,
                 plainto_tsquery('russian', %s) AS q(query)
            WHERE c.tsv @@ q.query AND d.status = 'indexed'
    """
    args += [top_k * 2, query]
    if doc_ids:
        sql += " AND c.doc_id = ANY(%s)"
        args.append(doc_ids)
    sql += """
            LIMIT %s
        )
        SELECT c.id, c.doc_id, c.text, c.metadata, d.filename,
               COALESCE(1.0/( %s + v.rank), 0) + COALESCE(1.0/( %s + f.rank), 0) AS rrf_score,
               1 - (c.embedding <=> %s::vector) AS vec_score
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        LEFT JOIN vec v ON v.id = c.id
        LEFT JOIN fts f ON f.id = c.id
        WHERE v.id IS NOT NULL OR f.id IS NOT NULL
    """
    args += [top_k * 2, RRF_K, RRF_K, vec]
    sql += " ORDER BY rrf_score DESC LIMIT %s"
    args.append(top_k)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    return [
        {
            **_row_to_dict((row[0], row[1], row[2], row[3], row[4], row[6])),
            "rrf_score": float(row[5]),
        }
        for row in rows
    ]


def get_chunks_by_ids(conn: psycopg.Connection, ids: list[str]) -> dict[str, dict]:
    """chunk_id → {doc_id, text, metadata}. Для валидации источников."""
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, doc_id, text, metadata FROM chunks WHERE id = ANY(%s)", (ids,)
        )
        rows = cur.fetchall()
    return {
        row[0]: {"doc_id": row[1], "text": row[2], "metadata": _as_dict(row[3])}
        for row in rows
    }


def get_chunk(conn: psycopg.Connection, chunk_id: str) -> dict | None:
    """Один фрагмент по id: {chunk_id, doc_id, text, metadata}. None, если нет."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, doc_id, text, metadata FROM chunks WHERE id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "chunk_id": row[0],
        "doc_id": row[1],
        "text": row[2],
        "metadata": _as_dict(row[3]),
    }


def _as_dict(value) -> dict:
    """JSONB из psycopg приезжает dict'ом; всё прочее считаем пустым."""
    return value if isinstance(value, dict) else {}


def _row_to_dict(row) -> dict:
    """Строка поиска → словарь результата с float-скором."""
    return {
        "chunk_id": row[0],
        "doc_id": row[1],
        "text": row[2],
        "metadata": row[3] if isinstance(row[3], dict) else {},
        "filename": row[4],
        "score": float(row[5]) if row[5] is not None else None,
    }
