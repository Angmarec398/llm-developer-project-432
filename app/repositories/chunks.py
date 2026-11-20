"""Repository-слой: таблица chunks в Postgres.

Фрагменты документа вставляются одным пакетом после разбора;
seq — порядок фрагмента в документе, назначается здесь по позиции
в списке, чтобы вызывающий код о нём не думал.

Шаг 6: векторная и полнотекстовая колонки. Эмбеддинги пишутся
пачкой; поиск — три режима. По векторам: косинусная близость,
порог отсекает семантический мусор. По словам: ранжирование
ts_rank, порог отсекает слабые совпадения. Гибрид: слияние двух
списков по обратным позициям (RRF) — фрагмент получает вклад
1/(60+ранг) от каждого списка, где найден, поэтому найденный
обоими оказывается выше; порог фильтрует здесь только векторную
половину — точное совпадение термина само по себе релевантно.

Шаг 7: get_chunk — один фрагмент по id для проверки ссылок
карточки (app/services/citations.py); фильтр по метаданным
принимает список значений (документы выбранные для генерации);
missing_embeddings_for_documents — досчёт векторов по выбранным
документам.
"""
import psycopg
from pgvector import Vector

_CHUNK_COLUMNS = "id, document_id, seq, text, page, section, article, brand"
_FILTER_KEYS = ("document_id", "page", "section", "article", "brand")


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


def save_embeddings(
    conn: psycopg.Connection,
    chunk_ids: list[int],
    vectors: list[list[float]],
) -> int:
    """Массово записать эмбеддинги фрагментов, вернуть число записанных.

    chunk_ids и vectors идут попарно: i-й вектор — эмбеддинг i-го
    фрагмента. Одним executemany — векторы приходят и уходят пачкой.
    """
    if len(chunk_ids) != len(vectors):
        raise ValueError("число идентификаторов не равно числу векторов")
    rows = [(Vector(vector), chunk_id)
            for chunk_id, vector in zip(chunk_ids, vectors)]
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s WHERE id = %s",
            rows,
        )
    return len(rows)


def missing_embeddings(
    conn: psycopg.Connection,
    *,
    limit: int,
) -> list[tuple[int, str]]:
    """Вернуть до limit фрагментов без вектора: пары (id, текст).

    Порядок — по возрастанию id: команда досчёта обходит таблицу
    стабильно, и повторный запуск ничего не пересчитывает.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, text FROM chunks
            WHERE embedding IS NULL
            ORDER BY id
            LIMIT %(limit)s
            """,
            {"limit": limit},
        )
        return list(cur.fetchall())


def missing_embeddings_for_documents(
    conn: psycopg.Connection,
    document_ids: list[str],
    *,
    limit: int,
) -> list[tuple[int, str]]:
    """Фрагменты без вектора из заданных документов: пары (id, текст).

    Тот же стабильный порядок по id, что и у missing_embeddings,
    но выборка ограничена списком документов — индексация внутри
    задачи генерации не трогает чужие документы.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, text FROM chunks
            WHERE embedding IS NULL
              AND document_id = ANY(%(document_ids)s)
            ORDER BY id
            LIMIT %(limit)s
            """,
            {"document_ids": document_ids, "limit": limit},
        )
        return list(cur.fetchall())


def count_missing_embeddings(conn: psycopg.Connection) -> int:
    """Сколько фрагментов ещё без вектора — для прогресса досчёта."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL")
        return cur.fetchone()[0]


def list_chunk_ids(conn: psycopg.Connection, document_id: str) -> list[int]:
    """Идентификаторы фрагментов документа в порядке seq.

    Порядок совпадает с порядком, в котором фрагменты сохранялись,
    — по нему эмбеддинги встают на свои места попарно.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM chunks WHERE document_id = %s ORDER BY seq",
            (document_id,),
        )
        return [row[0] for row in cur.fetchall()]


def get_chunk(conn: psycopg.Connection, chunk_id: int) -> dict | None:
    """Один фрагмент по id; None, если фрагмента с таким id нет.

    Без score — это не результат поиска, а проверка существования
    для верификации ссылок карточки.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return _chunk_dict(row)


def search_vector(
    conn: psycopg.Connection,
    query_vector: list[float],
    *,
    limit: int,
    min_score: float,
    filters: dict | None = None,
) -> list[dict]:
    """Поиск по смыслу: ближайшие к вектору запроса фрагменты.

    Оценка — косинусная близость (1 − cosine distance); фрагменты
    ниже порога min_score не возвращаются: мусор в контексте дороже
    пропуска релевантного.
    """
    params = {
        "query_vector": Vector(query_vector),
        "min_score": min_score,
        "limit": limit,
        **_filter_params(filters),
    }
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_CHUNK_COLUMNS}, score
            FROM (
                SELECT {_CHUNK_COLUMNS},
                       1 - (embedding <=> %(query_vector)s) AS score
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> %(query_vector)s) >= %(min_score)s
                  {_metadata_conditions(filters)}
            ) AS scored
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            params,
        )
        return [_to_chunk(row) for row in cur.fetchall()]


def search_fulltext(
    conn: psycopg.Connection,
    query: str,
    *,
    limit: int,
    min_score: float,
    filters: dict | None = None,
) -> list[dict]:
    """Поиск по словам: обычное ранжирование полнотекстового поиска.

    Запрос превращается в tsquery; оценка — ts_rank, фрагменты
    ниже порога min_score не возвращаются.
    """
    params = {
        "query": query,
        "min_score": min_score,
        "limit": limit,
        **_filter_params(filters),
    }
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_CHUNK_COLUMNS}, score
            FROM (
                SELECT {_CHUNK_COLUMNS},
                       ts_rank(tsv, search_query) AS score
                FROM chunks, plainto_tsquery('russian', %(query)s) AS search_query
                WHERE tsv @@ search_query
                  {_metadata_conditions(filters)}
            ) AS scored
            WHERE score >= %(min_score)s
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            params,
        )
        return [_to_chunk(row) for row in cur.fetchall()]


def search_hybrid(
    conn: psycopg.Connection,
    query_vector: list[float],
    query: str,
    *,
    limit: int,
    min_score: float,
    filters: dict | None = None,
) -> list[dict]:
    """Гибридный поиск: слияние векторного и полнотекстового списков.

    Каждый фрагмент получает вклад 1/(60+ранг) от каждого списка,
    где встретился, — слияние по обратным позициям (RRF), поэтому
    фрагмент, найденный обоими способами, оказывается выше любого,
    найденного одним. Порог фильтрует только векторную половину:
    полнотекстовый кандидат — точное совпадение терминов запроса,
    он релевантен по построению (так гибрид находит и артикулы,
    которые по смыслу промахиваются). Оценка результата — суммарный
    вклад RRF.
    """
    params = {
        "query_vector": Vector(query_vector),
        "query": query,
        "min_score": min_score,
        "limit": limit,
        **_filter_params(filters),
    }
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH vec AS (
                SELECT {_CHUNK_COLUMNS},
                       1 - (embedding <=> %(query_vector)s) AS vec_score,
                       row_number() OVER (
                           ORDER BY embedding <=> %(query_vector)s
                       ) AS vec_rank
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> %(query_vector)s) >= %(min_score)s
                  {_metadata_conditions(filters)}
            ),
            fts AS (
                SELECT {_CHUNK_COLUMNS},
                       ts_rank(tsv, search_query) AS fts_score,
                       row_number() OVER (
                           ORDER BY ts_rank(tsv, search_query) DESC
                       ) AS fts_rank
                FROM chunks, plainto_tsquery('russian', %(query)s) AS search_query
                WHERE tsv @@ search_query
                  {_metadata_conditions(filters)}
            ),
            merged AS (
                SELECT coalesce(v.id, f.id) AS id,
                       coalesce(v.document_id, f.document_id) AS document_id,
                       coalesce(v.seq, f.seq) AS seq,
                       coalesce(v.text, f.text) AS text,
                       coalesce(v.page, f.page) AS page,
                       coalesce(v.section, f.section) AS section,
                       coalesce(v.article, f.article) AS article,
                       coalesce(v.brand, f.brand) AS brand,
                        ((coalesce(1.0 / (60 + v.vec_rank), 0.0)
                         + coalesce(1.0 / (60 + f.fts_rank), 0.0)))::double precision AS score

                FROM vec AS v
                FULL OUTER JOIN fts AS f ON f.id = v.id
            )
            SELECT id, document_id, seq, text, page, section, article, brand, score
            FROM merged
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            params,
        )
        return [_to_chunk(row) for row in cur.fetchall()]


def _metadata_conditions(filters: dict | None) -> str:
    """Собрать SQL-условия по метаданным.

    Скалярное значение — `AND key = %(key)s`; список —
    `AND key = ANY(%(key)s)` (psycopg адаптирует list как массив):
    так фильтр по документам принимает список выбранных документов.
    """
    if not filters:
        return ""
    conditions = []
    for key in _FILTER_KEYS:
        value = filters.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            conditions.append(f"AND {key} = ANY(%({key})s)")
        else:
            conditions.append(f"AND {key} = %({key})s")
    if not conditions:
        return ""
    return "\n                  " + "\n                  ".join(conditions)


def _filter_params(filters: dict | None) -> dict:
    """Взять из фильтров только заданные значения метаданных."""
    if not filters:
        return {}
    return {
        key: value
        for key, value in filters.items()
        if key in _FILTER_KEYS and value is not None
    }


def _chunk_dict(row: tuple) -> dict:
    """dict фрагмента из кортежа колонок _CHUNK_COLUMNS."""
    return {
        "id": row[0],
        "document_id": row[1],
        "seq": row[2],
        "text": row[3],
        "page": row[4],
        "section": row[5],
        "article": row[6],
        "brand": row[7],
    }


def _to_chunk(row: tuple) -> dict:
    """Превратить строку результата поиска (колонки + score) в dict."""
    chunk = _chunk_dict(row)
    chunk["score"] = row[8]
    return chunk
