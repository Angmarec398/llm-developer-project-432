"""Тесты поиска по фрагментам на заранее заданных векторах.

Модель эмбеддингов НЕ загружается: векторы строятся аналитически —
единичные векторы с известной косинусной близостью к базовому
[1, 0, 0, ...]. Нужен поднятый postgres с применёнными миграциями
(make infra && make migrate); без базы тесты пропускаются.
"""
import math
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.config import settings
from app.core.db import connection
from app.rag.context import format_context
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo


@pytest.fixture()
def conn():
    """Соединение с живой БД; тест пропускается, если база недоступна."""
    try:
        with connection() as db_connection:
            yield db_connection
    except (psycopg.Error, PoolTimeout):
        pytest.skip("postgres недоступен (поднимите: make infra)")


@pytest.fixture()
def document_id(conn):
    """Тестовый документ; удаляется по завершении (фрагменты — каскадом)."""
    doc_id, _created = documents_repo.create_document(
        conn,
        filename="retrieval-test",
        file_path="data/retrieval-test",
        content_hash=uuid4().hex,
    )
    yield doc_id
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


def _query_vector() -> list[float]:
    """Базовый вектор запроса: [1, 0, 0, ...]."""
    vector = [0.0] * settings.embedding_dimension
    vector[0] = 1.0
    return vector


def _vector_with_similarity(cosine: float) -> list[float]:
    """Единичный вектор с заданной косинусной близостью к базовому.

    Косинусная близость к [1, 0, 0, ...] у [c, s, 0, ...] равна c —
    порог в тестах задаётся точно, без зависимости от модели.
    """
    sine = math.sqrt(1 - cosine**2)
    vector = [0.0] * settings.embedding_dimension
    vector[0] = cosine
    vector[1] = sine
    return vector


def _insert_chunks(
    conn: psycopg.Connection,
    document_id: str,
    specs: list[tuple[str, str, list[float] | None, dict]],
) -> dict[str, int]:
    """Вставить фрагменты по спецификациям, вернуть сопоставление имя → id.

    spec — кортеж (имя, текст, вектор или None, метаданные). Векторы
    вешаются пачкой через save_embeddings, порядок совпадает с seq.
    """
    chunk_dicts = [
        {"text": text, **metadata}
        for _name, text, _vector, metadata in specs
    ]
    chunks_repo.save_chunks(conn, document_id, chunk_dicts)
    ids = chunks_repo.list_chunk_ids(conn, document_id)
    assert len(ids) == len(specs)
    named_ids = {
        name: chunk_id
        for (name, _text, _vector, _metadata), chunk_id in zip(specs, ids)
    }
    chunk_ids = []
    vectors = []
    for (_name, _text, vector, _metadata), chunk_id in zip(specs, ids):
        if vector is not None:
            chunk_ids.append(chunk_id)
            vectors.append(vector)
    if chunk_ids:
        chunks_repo.save_embeddings(conn, chunk_ids, vectors)
    return named_ids


def test_vector_search_orders_by_similarity_and_threshold(conn, document_id):
    """Векторный поиск: ближе — выше; порог отсекает дальние фрагменты."""
    ids = _insert_chunks(conn, document_id, [
        ("far", "Габариты и вес", _vector_with_similarity(0.4), {}),
        ("exact", "Мощность двигателя 800 Вт", _query_vector(), {}),
        ("close", "Потребляемая мощность 800 ватт", _vector_with_similarity(0.99), {}),
        ("no_vector", "Фрагмент без вектора", None, {}),
    ])
    results = chunks_repo.search_vector(
        conn, _query_vector(), limit=10, min_score=0.5,
        filters={"document_id": document_id},
    )
    assert [row["id"] for row in results] == [ids["exact"], ids["close"]]
    strict = chunks_repo.search_vector(
        conn, _query_vector(), limit=10, min_score=0.999,
        filters={"document_id": document_id},
    )
    assert [row["id"] for row in strict] == [ids["exact"]]


def test_fulltext_search_finds_exact_term(conn, document_id):
    """Поиск по словам находит артикул буква в букву."""
    ids = _insert_chunks(conn, document_id, [
        ("spec", "Артикул KTL-1700, бренд KitchenLine", None, {"article": "KTL-1700"}),
        ("noise", "Чайник электрический, объём 1.7 л", None, {}),
    ])
    results = chunks_repo.search_fulltext(conn, "KTL-1700", limit=10, min_score=0.0)
    found_ids = {row["id"] for row in results}
    assert ids["spec"] in found_ids
    assert ids["noise"] not in found_ids


def test_hybrid_returns_union_and_boosts_dual_match(conn, document_id):
    """Гибрид: объединение списков; фрагмент, найденный обоими, — выше всех."""
    ids = _insert_chunks(conn, document_id, [
        ("dual", "Артикул KTL-1700, мощность 800 Вт", _vector_with_similarity(0.99), {}),
        ("vector_only", "Мощность двигателя 800 Вт", _query_vector(), {}),
        ("words_only", "Артикул KTL-1700, цвет чёрный", _vector_with_similarity(0.0), {}),
    ])
    results = chunks_repo.search_hybrid(
        conn, _query_vector(), "KTL-1700", limit=10, min_score=0.5,
        filters={"document_id": document_id},
    )
    found_ids = {row["id"] for row in results}
    assert found_ids == {ids["dual"], ids["vector_only"], ids["words_only"]}
    assert results[0]["id"] == ids["dual"]
    # words_only по смыслу промахивается: по одним векторам его нет...
    vector_results = chunks_repo.search_vector(
        conn, _query_vector(), limit=10, min_score=0.5,
        filters={"document_id": document_id},
    )
    assert ids["words_only"] not in {row["id"] for row in vector_results}
    # ...но гибрид спасает: точное совпадение термина релевантно по построению.


def test_metadata_filters_narrow_search(conn, document_id):
    """Фильтр по метаданным оставляет только фрагменты заданного документа."""
    ids = _insert_chunks(conn, document_id, [
        ("ours", "Мощность 800 Вт", _query_vector(), {"brand": "KitchenLine"}),
    ])
    other_document, _created = documents_repo.create_document(
        conn,
        filename="retrieval-other",
        file_path="data/retrieval-other",
        content_hash=uuid4().hex,
    )
    try:
        _insert_chunks(conn, other_document, [
            ("theirs", "Мощность 800 Вт", _query_vector(), {}),
        ])
        results = chunks_repo.search_vector(
            conn, _query_vector(), limit=10, min_score=0.5,
            filters={"document_id": document_id},
        )
        assert [row["id"] for row in results] == [ids["ours"]]
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (other_document,))


def test_missing_embeddings_lists_only_chunks_to_backfill(conn, document_id):
    """Досчёт видит только фрагменты без вектора; после записи — пусто."""
    ids = _insert_chunks(conn, document_id, [
        ("with_vector", "Мощность 800 Вт", _query_vector(), {}),
        ("without_vector", "Габариты и вес", None, {}),
    ])
    missing_ids = {chunk_id for chunk_id, _text in chunks_repo.missing_embeddings(conn, limit=100)}
    assert ids["without_vector"] in missing_ids
    assert ids["with_vector"] not in missing_ids
    assert chunks_repo.count_missing_embeddings(conn) == 1
    chunks_repo.save_embeddings(conn, [ids["without_vector"]], [_query_vector()])
    assert chunks_repo.count_missing_embeddings(conn) == 0


def test_format_context_includes_source_and_text():
    """Контекст содержит источник и текст; пустой поиск — пустая строка."""
    result = format_context([
        {"document_id": "doc1", "seq": 2, "text": "Мощность 800 Вт",
         "page": 3, "section": "Характеристики", "article": "KTL-1700",
         "brand": None, "score": 0.9}
    ])
    assert "документ doc1" in result
    assert "стр. 3" in result
    assert "артикул KTL-1700" in result
    assert "Мощность 800 Вт" in result
    assert format_context([]) == ""
