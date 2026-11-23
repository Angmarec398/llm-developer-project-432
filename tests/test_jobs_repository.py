"""Тесты репозитория jobs на живой базе (нужен поднятый postgres)."""
import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.db import connection
from app.repositories import jobs as jobs_repo
from app.schemas.cards import (
    CardDraft,
    GuardrailReport,
    MaskedItem,
    Seo,
    SourceRef,
)


@pytest.fixture()
def conn():
    """Соединение с живой БД; тест пропускается, если база недоступна."""
    try:
        with connection() as db_connection:
            yield db_connection
    except (psycopg.Error, PoolTimeout):
        pytest.skip("postgres недоступен (поднимите: make infra)")


@pytest.fixture()
def job_ids(conn):
    """Реестр созданных в тесте задач: удаляет их по завершении."""
    created: list[str] = []
    yield created
    with conn.cursor() as cur:
        for job_id in created:
            cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))


def test_create_job_is_idempotent(conn, job_ids):
    """Повтор с тем же ключом возвращает ту же задачу (created=False)."""
    job_id, created = jobs_repo.create_job(conn, {"supplier_text": "x"}, "test-idem-1")
    job_ids.append(job_id)
    again_id, again_created = jobs_repo.create_job(
        conn, {"supplier_text": "y"}, "test-idem-1"
    )
    assert created is True
    assert again_created is False
    assert again_id == job_id


def test_create_job_without_key(conn, job_ids):
    """Без ключа каждый вызов создаёт новую задачу."""
    first, first_created = jobs_repo.create_job(conn, {"supplier_text": "x"}, None)
    second, second_created = jobs_repo.create_job(conn, {"supplier_text": "x"}, None)
    job_ids.extend([first, second])
    assert first_created and second_created
    assert first != second


def test_set_status_and_get_job(conn, job_ids):
    """Статус, счётчик попыток и результат сохраняются и читаются обратно."""
    job_id, _ = jobs_repo.create_job(conn, {"supplier_text": "x"}, None)
    job_ids.append(job_id)
    jobs_repo.set_status(conn, job_id, "generating", bump_attempts=True)
    card = CardDraft(
        title="Чайник 2200 Вт",
        description="Описание",
        seo=Seo(title="Чайник"),
        sources=[SourceRef(chunk_id="abc123def456", doc_id="doc1")],
        confidence=0.9,
    )
    jobs_repo.set_status(conn, job_id, "approved", result=card)
    job = jobs_repo.get_job(conn, job_id)
    assert job["status"] == "approved"
    assert job["attempts"] == 1
    assert job["result"]["title"] == "Чайник 2200 Вт"
    assert job["result"]["confidence"] == 0.9


def test_get_job_missing(conn):
    """Запрос несуществующей задачи возвращает None."""
    assert jobs_repo.get_job(conn, "no-such-id") is None


def test_set_status_stores_guardrail_report(conn, job_ids):
    """Отчёт защиты сохраняется в задаче и читается обратно."""
    job_id, _ = jobs_repo.create_job(conn, {"supplier_text": "x"}, None)
    job_ids.append(job_id)
    report = GuardrailReport(
        masked=[MaskedItem(category="phone", label="[PHONE_1]")]
    )
    jobs_repo.set_status(conn, job_id, "done", guardrail_report=report)
    job = jobs_repo.get_job(conn, job_id)
    assert job["guardrail_report"]["masked"][0]["category"] == "phone"
    assert job["guardrail_report"]["masked"][0]["label"] == "[PHONE_1]"
