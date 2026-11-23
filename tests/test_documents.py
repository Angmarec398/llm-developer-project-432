"""Сквозные тесты шага 5: сервис разбора и API документов.

Ходят в живую базу (как test_jobs_repository): документы создаются
перед тестом и удаляются после — фрагменты уходят каскадом.
Запуск процесса разбора (Temporal) подменяется фейковым клиентом,
сам разбор проверяется вызовом сервиса напрямую.
"""
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.db import connection
from app.main import app
from app.repositories import documents as documents_repo
from app.routers import documents as documents_router
from app.services.documents import process_document

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def document():
    """Создать документ в БД и удалить его после теста."""
    created_ids: list[str] = []

    def factory(filename: str) -> tuple[str, str]:
        file_path = DATA_DIR / filename
        content_hash = f"test-{filename}-{len(created_ids)}"
        with connection() as conn:
            document_id, _ = documents_repo.create_document(
                conn, filename, str(file_path), content_hash
            )
        created_ids.append(document_id)
        return document_id, str(file_path)

    yield factory
    with connection() as conn, conn.cursor() as cur:
        for document_id in created_ids:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def test_process_document_indexes_spec(document, monkeypatch):
    """Спецификация доходит до indexed: строки целые, артикулы в метаданных."""
    import app.rag.reindex as reindex_mod

    def fake_index_pending(conn, doc_id=None, batch=256):
        """Эмбеддинги в тесте не считаем: модель не грузится."""
        return 0

    monkeypatch.setattr(reindex_mod, "index_pending_chunks", fake_index_pending)
    document_id, file_path = document("kettle_spec.xlsx")
    saved = process_document(document_id, file_path)
    assert saved > 0
    with connection() as conn:
        result = documents_repo.get_document(conn, document_id)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT text, metadata FROM chunks WHERE doc_id = %s ORDER BY ordinal",
                (document_id,),
            )
            rows = cur.fetchall()
    assert result["status"] == "indexed"
    assert result["chunks_count"] == saved
    first_text, first_meta = rows[0]
    assert first_meta["articul"]  # артикул поднят в метаданные
    for line in first_text.split("\n"):
        assert ": " in line  # «артикул: параметр: значение», строка не разорвана


def test_process_document_fails_scan_with_reason(document):
    """Отсканированный pdf получает отказ с причиной, а не пустой результат."""
    document_id, file_path = document("boiler_scan.pdf")
    saved = process_document(document_id, file_path)
    assert saved == 0
    with connection() as conn:
        result = documents_repo.get_document(conn, document_id)
    assert result["status"] == "failed"
    assert result["error"]
    assert result["chunks_count"] == 0


def test_reupload_creates_no_duplicates():
    """Повторная загрузка того же содержимого возвращает существующий документ."""
    with connection() as conn:
        first_id, created_first = documents_repo.create_document(
            conn, "a.pdf", "/tmp/a.pdf", "test-dedup-hash"
        )
        second_id, created_second = documents_repo.create_document(
            conn, "a.pdf", "/tmp/a.pdf", "test-dedup-hash"
        )
        conn.cursor().execute("DELETE FROM documents WHERE id = %s", (first_id,))
    assert created_first is True
    assert created_second is False
    assert second_id == first_id


class _FakeTemporalClient:
    """Фейк Temporal-клиента: запоминает запуски процессов, сети нет."""

    def __init__(self) -> None:
        self.started: list[dict] = []

    async def start_workflow(self, workflow, args, id, task_queue):
        """Записать параметры запуска процесса вместо обращения к движку."""
        self.started.append(
            {"workflow": workflow, "args": args, "id": id, "task_queue": task_queue}
        )


async def test_upload_accepts_file_and_starts_workflow(monkeypatch, tmp_path):
    """POST /documents отвечает 202 с id и запускает DocumentWorkflow."""
    fake_client = _FakeTemporalClient()

    async def fake_get_client():
        """Вернуть фейковый Temporal-клиент без обращения к сети."""
        return fake_client

    monkeypatch.setattr(documents_router, "get_client", fake_get_client)
    monkeypatch.setattr(documents_router.settings, "upload_dir", str(tmp_path))
    unique_content = uuid.uuid4().hex.encode()  # иначе сработает дедупликация
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/documents",
            files={"file": ("uploaded_spec.xlsx", unique_content)},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["id"]
        assert len(fake_client.started) == 1
        assert fake_client.started[0]["workflow"] == "DocumentWorkflow"
        assert fake_client.started[0]["id"] == f"document-{body['id']}"

        status_response = await client.get(f"/documents/{body['id']}")
        assert status_response.status_code == 200
        assert status_response.json()["chunks_count"] == 0  # разбор не запускался

        with connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (body["id"],))


async def test_upload_rejects_bad_extension(monkeypatch, tmp_path):
    """Файл с недопустимым расширением отклоняется с 415."""
    monkeypatch.setattr(documents_router.settings, "upload_dir", str(tmp_path))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/documents", files={"file": ("notes.txt", b"hello")}
        )
    assert response.status_code == 415


async def test_get_unknown_document_404():
    """Состояние несуществующего документа — 404."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/documents/no-such-id")
    assert response.status_code == 404
