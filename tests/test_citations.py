"""Тесты проверки ссылок: фрагмент существует в базе и был в контексте.

Нужна живая база (make infra && make migrate); без неё тесты пропускаются.
"""
import uuid

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.db import connection
from app.repositories import documents as documents_repo
from app.schemas.cards import CardDraft, SourceRef
from app.services import citations


@pytest.fixture()
def conn():
    """Соединение с живой БД; тест пропускается, если база недоступна."""
    try:
        with connection() as db_connection:
            yield db_connection
    except (psycopg.Error, PoolTimeout):
        pytest.skip("postgres недоступен (поднимите: make infra)")


@pytest.fixture()
def document(conn):
    """Документ с двумя фрагментами в базе; чистится после теста."""
    doc_id, _created = documents_repo.create_document(
        conn, "cite-test", "data/cite-test", uuid.uuid4().hex
    )
    with conn.cursor() as cur:  # зачистка после возможного прошлого падения
        cur.execute("DELETE FROM chunks WHERE id IN ('citec1', 'citec2')")
    documents_repo.insert_chunks(conn, [
        {
            "id": "citec1",
            "text": "Мощность 800 Вт",
            "ordinal": 0,
            "metadata": {"doc_id": doc_id, "page": 1},
        },
        {
            "id": "citec2",
            "text": "Гарантия 24 месяца",
            "ordinal": 1,
            "metadata": {"doc_id": doc_id, "page": 1},
        },
    ])
    yield doc_id
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


def _card(*chunk_ids: str) -> CardDraft:
    """Карточка, ссылающаяся на заданные фрагменты."""
    return CardDraft(
        title="Чайник 2200 Вт",
        description="Описание",
        sources=[SourceRef(chunk_id=chunk_id) for chunk_id in chunk_ids],
    )


def test_valid_citations_pass(conn, document):
    """Ссылки на фрагменты из контекста — без нарушений."""
    violations = citations.verify_citations(
        conn, _card("citec1", "citec2"), ["citec1", "citec2"]
    )
    assert violations == []


def test_unknown_chunk_rejected(conn, document):
    """Ссылка на несуществующий фрагмент — нарушение."""
    violations = citations.verify_citations(conn, _card("ghost000000"), ["citec1"])
    assert any("не существует" in violation for violation in violations)


def test_out_of_context_chunk_rejected(conn, document):
    """Фрагмент есть в базе, но не был в контексте — нарушение."""
    violations = citations.verify_citations(conn, _card("citec2"), ["citec1"])
    assert any("не входил" in violation for violation in violations)


def test_duplicate_citation_checked_once(conn, document):
    """Повторная ссылка на тот же битый фрагмент — одно нарушение."""
    violations = citations.verify_citations(
        conn, _card("ghost000000", "ghost000000"), ["citec1"]
    )
    assert len(violations) == 1


def test_card_without_sources_passes(conn, document):
    """Карточка без ссылок не нарушает ничего."""
    assert citations.verify_citations(conn, _card(), ["citec1"]) == []
