"""Тесты проверки ссылок карточки на живой базе (нужен postgres).

Сценарии: валидные ссылки; ссылка на несуществующий фрагмент;
ссылка на существующий, но не поданный в контекст фрагмент
(выдумывание с правдоподобным видом); смешанный случай с дублем.
"""
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.db import connection
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas.cards import CardSeo, CardSource, ProductCard
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
def chunk_ids(conn):
    """Два тестовых фрагмента; документ удаляется по завершении."""
    doc_id, _created = documents_repo.create_document(
        conn,
        filename="citations-test",
        file_path="data/citations-test",
        content_hash=uuid4().hex,
    )
    chunks_repo.save_chunks(conn, doc_id, [
        {"text": "Мощность 2200 Вт"},
        {"text": "Объём 1.7 л"},
    ])
    ids = chunks_repo.list_chunk_ids(conn, doc_id)
    yield ids
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


def _free_chunk_id(conn) -> int:
    """Идентификатор, которого точно нет в таблице фрагментов."""
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(max(id), 0) + 1000000 FROM chunks")
        return cur.fetchone()[0]


def _card(*source_chunk_ids: int) -> ProductCard:
    """Карточка, ссылающаяся на заданные фрагменты."""
    return ProductCard(
        title="Чайник",
        description="Описание",
        seo=CardSeo(slug="chainik"),
        sources=[
            CardSource(chunk_id=chunk_id, document="citations-test")
            for chunk_id in source_chunk_ids
        ],
        confidence=0.9,
    )


def test_valid_citations_pass(conn, chunk_ids):
    """Ссылки на существующие фрагменты из контекста — нарушений нет."""
    card = _card(chunk_ids[0], chunk_ids[1])
    violations = citations.verify_citations(
        conn, card, context_chunk_ids=chunk_ids
    )
    assert violations == []


def test_unknown_chunk_is_violation(conn, chunk_ids):
    """Ссылка на несуществующий фрагмент — нарушение."""
    card = _card(_free_chunk_id(conn))
    violations = citations.verify_citations(
        conn, card, context_chunk_ids=chunk_ids
    )
    assert len(violations) == 1
    assert "не существует" in violations[0]


def test_existing_chunk_outside_context_is_violation(conn, chunk_ids):
    """Ссылка на существующий, но не поданный фрагмент — выдумывание."""
    card = _card(chunk_ids[0])
    violations = citations.verify_citations(
        conn, card, context_chunk_ids=[chunk_ids[1]]
    )
    assert len(violations) == 1
    assert "не входил" in violations[0]


def test_mixed_sources_report_each_problem_once(conn, chunk_ids):
    """Строка нарушения на каждую битую ссылку; дубль не дублируется."""
    card = _card(chunk_ids[0], chunk_ids[0], _free_chunk_id(conn))
    violations = citations.verify_citations(
        conn, card, context_chunk_ids=[chunk_ids[1]]
    )
    assert len(violations) == 2  # валидная ссылка и её дубль — один раз
