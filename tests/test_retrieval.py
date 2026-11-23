"""Тесты retrieval на фиксированных векторах: embedder подменён, модель
не грузится. Нужна живая база с миграциями (make infra && make migrate).
"""
import math
import uuid

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.db import connection
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo

DIM = 768  # размерность колонки embedding (миграция 0007)


def _unit(vector: list[float]) -> list[float]:
    """Нормированный вектор, добитый нулями до размерности колонки."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] + [0.0] * (DIM - len(vector))


E_BLENDER = _unit([1, 0])
E_KETTLE = _unit([0, 1])


@pytest.fixture()
def conn():
    """Соединение с живой БД; тест пропускается, если база недоступна."""
    try:
        with connection() as db_connection:
            yield db_connection
    except (psycopg.Error, PoolTimeout):
        pytest.skip("postgres недоступен (поднимите: make infra)")


@pytest.fixture()
def document(conn, monkeypatch):
    """Документ с фрагментами и векторами; embed_query подменён ключевыми словами."""
    doc_id, _created = documents_repo.create_document(
        conn, "retr-test.pdf", "data/retr-test.pdf", uuid.uuid4().hex, kind="pdf"
    )
    documents_repo.set_status(conn, doc_id, "indexed")
    rows = [
        ("retrc1", 0, "Мощность блендера 800 Вт", E_BLENDER),
        ("retrc2", 1, "Чайник 2200 Вт, объём 1.7 л", E_KETTLE),
        ("retrc3", 2, "Гарантия 24 месяца", _unit([1, 1])),
    ]
    documents_repo.insert_chunks(conn, [
        {
            "id": chunk_id,
            "text": text,
            "ordinal": ordinal,
            "metadata": {"doc_id": doc_id, "page": 1},
        }
        for chunk_id, ordinal, text, _vec in rows
    ])
    chunks_repo.upsert_embeddings(
        conn, [(chunk_id, vec) for chunk_id, _ord, _text, vec in rows]
    )

    def fake_embed_query(text: str) -> list[float]:
        """«блендер» → вектор блендера, всё прочее → вектор чайника."""
        return E_BLENDER if "блендер" in text.lower() else E_KETTLE

    monkeypatch.setattr(chunks_repo, "embed_query", fake_embed_query)
    yield doc_id
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


def test_vector_search_finds_semantic_match(conn, document):
    """Ближайший по косинусу фрагмент — первый в выдаче."""
    results = chunks_repo.search_vector(conn, "мощность блендера", top_k=3)
    assert results
    assert results[0]["chunk_id"] == "retrc1"
    assert results[0]["score"] >= 0.9


def test_vector_search_threshold_filters(conn, document):
    """Порог отсекает далёкие векторы."""
    results = chunks_repo.search_vector(
        conn, "мощность блендера", top_k=5, threshold=0.99
    )
    ids = [result["chunk_id"] for result in results]
    assert "retrc1" in ids
    assert "retrc2" not in ids


def test_fts_search_exact_words(conn, document):
    """Полнотекстовый поиск находит фрагмент по слову «гарантия»."""
    results = chunks_repo.search_fts(conn, "гарантия", top_k=50)
    assert any(result["chunk_id"] == "retrc3" for result in results)


def test_hybrid_combines_vector_and_fts(conn, document):
    """Гибридная выдача содержит векторное попадание первым."""
    results = chunks_repo.search_hybrid(conn, "мощность блендера", top_k=3)
    ids = [result["chunk_id"] for result in results]
    assert "retrc1" in ids
    assert results[0]["chunk_id"] == "retrc1"
    assert all("rrf_score" in result for result in results)


def test_doc_filter(conn, document):
    """Фильтр по чужому документу отсекает всю выдачу."""
    results = chunks_repo.search_vector(
        conn, "мощность блендера", top_k=3, doc_ids=["другой-doc"]
    )
    assert results == []


def test_get_chunks_by_ids(conn, document):
    """Чтение фрагментов по id — для проверки источников."""
    found = chunks_repo.get_chunks_by_ids(conn, ["retrc1", "retrc3"])
    assert set(found) == {"retrc1", "retrc3"}
    assert found["retrc1"]["text"].startswith("Мощность")
    assert found["retrc1"]["metadata"]["page"] == 1
