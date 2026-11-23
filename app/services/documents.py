"""Сервис ingestion: parse → normalize → chunk → save → embed.

Статусная машина документа: new → parsing → indexed | failed.
Скан-PDF без текстового слоя честно помечается failed: no text layer.
Сбой эмбеддингов не откатывает разбор: фрагменты уже в базе, документ
остаётся indexed, без вектора их доберёт команда досчёта (make reindex).
"""
from pathlib import Path

from app.core.db import connection
from app.core.logging import get_logger
from app.parsers import normalizer
from app.parsers.chunker import chunk_blocks
from app.parsers.docx import parse_docx
from app.parsers.pdf import PdfNoTextLayer, parse_pdf
from app.parsers.xlsx import parse_xlsx
from app.repositories import documents as docs_repo

log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx"}


def parse_to_chunks(content: bytes, kind: str, doc_id: str) -> list[dict]:
    """Файл → нормализованные чанки с метаданными. Без БД — чистая функция."""
    if kind == "pdf":
        blocks = parse_pdf(content)
    elif kind == "docx":
        blocks = parse_docx(content)
    elif kind == "xlsx":
        blocks = parse_xlsx(content)
    else:
        raise ValueError(f"неподдерживаемый формат: {kind}")
    blocks = normalizer.normalize(blocks)
    return chunk_blocks(blocks, doc_id)


def ingest(conn, doc_id: str, content: bytes, kind: str, *, embed: bool = True) -> int:
    """Полный ingestion одного документа. Возвращает число чанков.

    embed=True: после сохранения чанков сразу считать embeddings.
    Если эмбеддинг упал — документ остаётся indexed (чанки валидны),
    статус понижается до embed_failed: поиск его не увидит, пока
    не прогонят make reindex.
    """
    docs_repo.set_status(conn, doc_id, "parsing")
    try:
        chunks = parse_to_chunks(content, kind, doc_id)
    except PdfNoTextLayer:
        docs_repo.set_status(conn, doc_id, "failed", error="no text layer (скан? нужен OCR)")
        log.warning("document_no_text_layer", doc_id=doc_id)
        return 0

    docs_repo.delete_chunks(conn, doc_id)
    docs_repo.insert_chunks(conn, chunks)
    docs_repo.set_chunks_count(conn, doc_id, len(chunks))
    docs_repo.set_status(conn, doc_id, "indexed")
    log.info("document_indexed", doc_id=doc_id, kind=kind, chunks=len(chunks))

    if embed and chunks:
        try:
            from app.rag.reindex import index_pending_chunks

            indexed = index_pending_chunks(conn, doc_id)
            log.info("document_embedded", doc_id=doc_id, vectors=indexed)
        except Exception as exc:  # noqa: BLE001 — эмбеддинг не должен ронять разбор
            docs_repo.set_status(
                conn, doc_id, "embed_failed", error=f"embedding: {exc!r}"
            )
            log.error("document_embed_failed", doc_id=doc_id, error=repr(exc))
            # не роняем ingestion: чанки сохранены, эмбеддинги досчитает make reindex
    return len(chunks)


def process_document(document_id: str, file_path: str) -> int:
    """Вход для activity разбора: прочитать файл с диска и провести ingestion.

    Любая ошибка разбора превращается в статус failed с внятной причиной —
    исключение наружу не уходит, процесс разбора не падает.
    """
    path = Path(file_path)
    kind = ALLOWED_EXTENSIONS.get(path.suffix.lower())
    if kind is None:
        with connection() as conn:
            docs_repo.set_status(
                conn, document_id, "failed",
                error=f"неподдерживаемый формат: {path.suffix}",
            )
        return 0
    try:
        content = path.read_bytes()
        with connection() as conn:
            return ingest(conn, document_id, content, kind)
    except Exception as error:  # noqa: BLE001 — любая ошибка разбора — отказ документа
        with connection() as conn:
            docs_repo.set_status(conn, document_id, "failed", error=f"ошибка разбора: {error!r}")
        log.warning("document_failed", document_id=document_id, reason=repr(error))
        return 0
