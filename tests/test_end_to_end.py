"""Сквозной тест сценария шага 7: документы → поиск → генерация → проверка ссылок.

LLM подменён очередью заготовленных ответов, поиск — фиксированной
выдачей по фрагментам, заранее лежащим в тестовой базе: модель
эмбеддингов не грузится. Оркестрация и проверка ссылок идут по-
настоящему — нужен живой postgres (make infra && make migrate);
без базы тесты пропускаются.
"""
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.config import settings
from app.core.db import connection
from app.rag import retrieval
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas.cards import CardSeo, CardSource, CritiqueReport, ProductCard
from app.services import cards as cards_service
from app.services import structured


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
    """Документ с двумя фрагментами в тестовой базе; чистится после теста."""
    doc_id, _created = documents_repo.create_document(
        conn,
        filename="e2e-test",
        file_path="data/e2e-test",
        content_hash=uuid4().hex,
    )
    documents_repo.set_status(conn, doc_id, "indexed")
    chunks_repo.save_chunks(conn, doc_id, [
        {"text": "Чайник электрический, мощность 2200 Вт, объём 1.7 л"},
        {"text": "Корпус из нержавеющей стали, гарантия 2 года"},
    ])
    chunk_ids = chunks_repo.list_chunk_ids(conn, doc_id)
    chunks_repo.save_embeddings(conn, chunk_ids, [
        [0.0] * settings.embedding_dimension for _chunk_id in chunk_ids
    ])
    chunks = [chunks_repo.get_chunk(conn, chunk_id) for chunk_id in chunk_ids]
    yield {"id": doc_id, "chunks": chunks}
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


def _card(*source_chunk_ids: int, confidence: float = 0.9) -> ProductCard:
    """Карточка, ссылающаяся на заданные фрагменты."""
    return ProductCard(
        title="Чайник 2200 Вт",
        description="Чайник из нержавеющей стали, объём 1.7 л",
        characteristics={"мощность": "2200 Вт"},
        benefits=["Быстрый нагрев"],
        seo=CardSeo(slug="chainik-2200"),
        sources=[
            CardSource(chunk_id=chunk_id, document="e2e-test")
            for chunk_id in source_chunk_ids
        ],
        confidence=confidence,
    )


def _mock_llm(monkeypatch: pytest.MonkeyPatch, answers: list) -> list[str]:
    """Подменить клиент LLM очередью ответов; вернуть список промптов."""
    prompts: list[str] = []
    queue = iter(answers)

    async def fake_run_agent(agent, prompt):
        prompts.append(prompt)
        return next(queue)

    monkeypatch.setattr(structured, "run_agent", fake_run_agent)
    return prompts


def _stub_search(
    monkeypatch: pytest.MonkeyPatch, chunks: list[dict]
) -> list[dict]:
    """Подменить поиск фиксированной выдачей; вернуть список вызовов."""
    calls: list[dict] = []

    def fake_search(query, *, mode, limit=None, min_score=None, filters=None):
        calls.append(
            {"query": query, "mode": mode, "filters": filters}
        )
        return [dict(chunk, score=0.95) for chunk in chunks]

    monkeypatch.setattr(retrieval, "search", fake_search)
    return calls


async def test_full_cycle_approved(conn, document, monkeypatch):
    """Согласовано: поиск → генерация → проверка ссылок → критик approve."""
    good_card = _card(
        document["chunks"][0]["id"], document["chunks"][1]["id"]
    )
    prompts = _mock_llm(monkeypatch, [good_card, CritiqueReport(verdict="approve")])
    calls = _stub_search(monkeypatch, document["chunks"])

    card, attempts, verdict = await cards_service.run_card_pipeline(
        [document["id"]], hint="чайник"
    )

    assert verdict == "approved"
    assert attempts == 1
    assert card == good_card
    assert calls[0]["mode"] == "hybrid"  # гибридный режим
    assert calls[0]["filters"] == {"document_id": [document["id"]]}  # фильтр по выбранным
    assert calls[0]["query"] == "чайник"  # запрос из подсказки
    # контекст с опознаваемым фрагментом ушёл в промпт генерации
    assert f"id {document['chunks'][0]['id']}" in prompts[0]


async def test_fabricated_citation_triggers_rework(conn, document, monkeypatch):
    """Выдуманная ссылка (фрагмент есть, но не в контексте) — переделка."""
    first = _card(document["chunks"][1]["id"])  # поиск отдаёт только chunks[0]
    fixed = _card(document["chunks"][0]["id"])
    prompts = _mock_llm(monkeypatch, [
        first, fixed, CritiqueReport(verdict="approve"),
    ])
    _stub_search(monkeypatch, [document["chunks"][0]])

    card, attempts, verdict = await cards_service.run_card_pipeline(
        [document["id"]], hint="чайник"
    )

    assert verdict == "approved"
    assert attempts == 2
    assert card == fixed
    # текст ошибки ушёл во вторую генерацию как feedback
    assert "не входил в поданный контекст" in prompts[1]


async def test_repeated_fabrication_escalates_to_human(conn, document, monkeypatch):
    """Повторная выдуманная ссылка — эскалация к человеку, критик не нужен."""
    ghost_id = document["chunks"][0]["id"] + 1000000  # такого фрагмента нет
    fabricated = _card(ghost_id)
    prompts = _mock_llm(monkeypatch, [fabricated, fabricated])
    _stub_search(monkeypatch, document["chunks"])

    card, attempts, verdict = await cards_service.run_card_pipeline(
        [document["id"]], hint="чайник"
    )

    assert verdict == "citation_fraud"
    assert attempts == 2
    assert card == fabricated
    assert len(prompts) == 2  # критик не вызывался: ссылки битые оба раза


async def test_missing_document_is_a_clear_error(conn, monkeypatch):
    """Несуществующий документ — понятная ошибка, а не падение."""
    with pytest.raises(ValueError, match="не найден"):
        await cards_service.run_card_pipeline(["no-such-document"])
