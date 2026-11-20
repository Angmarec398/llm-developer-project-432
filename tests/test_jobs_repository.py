"""Тесты репозитория jobs на живой базе (нужен поднятый postgres)."""
import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.db import connection
from app.repositories import jobs as jobs_repo
from app.schemas.cards import CardSeo, CardSource, ProductCard


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
    card = ProductCard(
        title="Чайник",
        description="Описание",
        seo=CardSeo(slug="chainik"),
        sources=[CardSource(chunk_id=1, document="supplier_text")],
        confidence=0.9,
    )
    jobs_repo.set_status(conn, job_id, "approved", result=card)
    job = jobs_repo.get_job(conn, job_id)
    assert job["status"] == "approved"
    assert job["attempts"] == 1
    assert job["result"]["title"] == "Чайник"
    assert job["result"]["confidence"] == 0.9


def test_get_job_missing(conn):
    """Запрос несуществующей задачи возвращает None."""
    assert jobs_repo.get_job(conn, "no-such-id") is None
