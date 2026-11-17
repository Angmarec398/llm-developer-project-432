"""Загрузка документов поставщиков и чтение их состояния.

Обработчик загрузки сохраняет файл, создаёт документ и запускает
процесс разбора; воркер гоняет разбор и пишет фрагменты в базу.
Идемпотентность двойная: на уровне БД (content_hash — уникальный
ключ) и на уровне движка (workflow_id детерминированно выводится
из document_id — повторный старт с тем же id Temporal не выполнит).
"""
import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.db import connection
from app.repositories import documents as documents_repo
from app.temporal.client import get_client

router = APIRouter(tags=["documents"])


@router.post("/documents", status_code=202)
async def upload_document(file: UploadFile):
    """Принять файл, проверить расширение и размер, запустить разбор.

    Отвечает 202 и идентификатором документа. Повторная загрузка
    того же файла не создаёт дублей: возвращается id существующего
    документа, процесс разбора повторно не стартует.
    """
    extension = Path(file.filename or "").suffix.lower()
    if extension not in settings.upload_allowed_extensions:
        raise HTTPException(415, f"неподдерживаемый формат: {extension}")
    content = await file.read()
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(413, "файл слишком большой")

    content_hash = hashlib.sha256(content).hexdigest()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{content_hash}{extension}"
    file_path.write_bytes(content)

    with connection() as conn:
        document_id, created = documents_repo.create_document(
            conn, file.filename or file_path.name, str(file_path), content_hash
        )
        document = documents_repo.get_document(conn, document_id)
    if created:
        client = await get_client()
        await client.start_workflow(
            "DocumentWorkflow",
            args=[document_id, str(file_path)],
            id=f"document-{document_id}",
            task_queue=settings.temporal_task_queue,
        )
    return JSONResponse(document, status_code=202)


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Вернуть состояние документа со счётчиком фрагментов."""
    with connection() as conn:
        document = documents_repo.get_document(conn, document_id)
    if not document:
        raise HTTPException(404, "document not found")
    return document
