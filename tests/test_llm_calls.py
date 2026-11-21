"""Клиент модели пишет каждый успешный вызов в таблицу учёта (живая БД)."""
from decimal import Decimal
from types import SimpleNamespace

import psycopg
import pytest
from psycopg_pool import PoolTimeout

from app.core.config import settings
from app.core.context import bind_job, current_job_id
from app.core.db import connection
from app.llm import client as llm_client
from app.repositories import jobs as jobs_repo

_TEST_AGENT_NAME = "TestAccountingAgent"


@pytest.fixture()
def conn():
    """Соединение с живой БД; тест пропускается, если база недоступна."""
    try:
        with connection() as db_connection:
            yield db_connection
    except (psycopg.Error, PoolTimeout):
        pytest.skip("postgres недоступен (поднимите: make infra)")


@pytest.fixture()
def cleanup(conn):
    """Удаляет записи учёта и задачи теста по завершении."""
    yield
    with conn.cursor() as cur:
        cur.execute("DELETE FROM llm_calls WHERE agent = %s", (_TEST_AGENT_NAME,))
        cur.execute("DELETE FROM jobs WHERE idempotency_key = %s", ("test-llm-calls",))


def _fake_result(prompt_tokens: int, completion_tokens: int):
    """Подмена RunResult SDK: финальный ответ + usage одного ответа модели."""
    usage = SimpleNamespace(
        input_tokens=prompt_tokens, output_tokens=completion_tokens
    )
    return SimpleNamespace(
        final_output="ok", raw_responses=[SimpleNamespace(usage=usage)]
    )


def _fake_agent(model: str):
    """Минимальный агент: клиенту нужны только name и model."""
    return SimpleNamespace(name=_TEST_AGENT_NAME, model=model)


def _patch_runner(monkeypatch, prompt_tokens: int, completion_tokens: int):
    """Подменить Runner.run: сети нет, ответ — фиксированный usage."""

    async def fake_run(agent, prompt, **_kwargs):
        return _fake_result(prompt_tokens, completion_tokens)

    monkeypatch.setattr(llm_client.Runner, "run", fake_run)


def _fetch_test_calls(conn) -> list[dict]:
    """Строки учёта тестового агента."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT job_id, agent, model, prompt_tokens, completion_tokens,
                   cost, duration_ms
            FROM llm_calls
            WHERE agent = %s
            ORDER BY id
            """,
            (_TEST_AGENT_NAME,),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


async def test_call_recorded_with_usage_and_duration(conn, cleanup, monkeypatch):
    """После вызова в таблице: модель агента, токены, длительность, стоимость."""
    monkeypatch.setattr(settings, "llm_price_input_per_mtok", Decimal(2))
    monkeypatch.setattr(settings, "llm_price_output_per_mtok", Decimal(4))
    _patch_runner(monkeypatch, prompt_tokens=1500, completion_tokens=400)
    await llm_client.run_agent(_fake_agent(settings.llm_model), "привет")

    calls = _fetch_test_calls(conn)
    assert len(calls) == 1
    row = calls[0]
    assert row["model"] == settings.llm_model
    assert row["prompt_tokens"] == 1500
    assert row["completion_tokens"] == 400
    assert row["duration_ms"] >= 0
    # 1500 * 2 + 400 * 4 = 4600 за миллион токенов = 0.0046 (Decimal).
    assert row["cost"] == Decimal("0.004600")



async def test_job_id_from_context_and_utility_prices(
    conn, cleanup, monkeypatch
):
    """job_id в строке — из контекста; цены служебной модели свои."""
    monkeypatch.setattr(settings, "llm_utility_price_input_per_mtok", 3)
    monkeypatch.setattr(settings, "llm_utility_price_output_per_mtok", 9)
    _patch_runner(monkeypatch, prompt_tokens=1000, completion_tokens=500)
    job_id, _created = jobs_repo.create_job(conn, {"hint": "x"}, "test-llm-calls")

    with bind_job(job_id):
        await llm_client.run_agent(
            _fake_agent(settings.llm_model_utility), "привет"
        )

    calls = _fetch_test_calls(conn)
    assert len(calls) == 1
    row = calls[0]
    assert row["job_id"] == job_id
    assert row["model"] == settings.llm_model_utility
    # 1000 * 3 + 500 * 9 = 7500 за миллион токенов = 0.0075.
    assert row["cost"] == Decimal("0.007500")

def test_bind_job_is_scoped_to_block():
    """bind_job действует только внутри блока: соседние задачи не видят чужой id."""
    assert current_job_id() is None
    with bind_job("job-scoped"):
        assert current_job_id() == "job-scoped"
    assert current_job_id() is None
