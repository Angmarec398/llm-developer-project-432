"""Сервис документа: разбор файла поставщика в фрагменты и векторы.

Конвейер: разбор по формату → нормализация → резка → сохранение
→ эмбеддинги. Состояния документа переключаются по ходу:
new → parsing → indexed | failed.

Отказ разбора: молча дать пустой результат нельзя — пустая карточка
без объяснения выглядит как поломка сервиса, поэтому любая ошибка
разбора превращается в статус failed с внятной причиной. С эмбеддингами
наоборот: сбой не откатывает разбор — фрагменты уже в базе, документ
уже indexed, без вектора их доберёт команда досчёта (app/rag/reindex).
"""
from pathlib import Path

from app.core.config import settings
from app.core.db import connection
from app.core.logging import get_logger
from app.parsers.chunker import Chunk, chunk_blocks
from app.parsers.docx import parse_docx
from app.parsers.normalizer import normalize_blocks
from app.parsers.pdf import NoTextLayerError, parse_pdf
from app.parsers.xlsx import parse_xlsx
from app.rag import embedder
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo

log = get_logger(__name__)

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
}


def process_document(document_id: str, file_path: str) -> int:
    """Разобрать документ в фрагменты, посчитать векторы, сохранить всё.

    Возвращает число сохранённых фрагментов. При невозможности разбора
    ставит документу отказ с причиной и возвращает 0 — исключение
    наружу не уходит, процесс разбора не падает.
    """
    extension = Path(file_path).suffix.lower()
    parser = PARSERS.get(extension)
    if parser is None:
        _fail(document_id, f"неподдерживаемый формат: {extension}")
        return 0

    with connection() as conn:
        documents_repo.set_status(conn, document_id, "parsing")
    try:
        blocks = normalize_blocks(parser(file_path))
        chunks = chunk_blocks(blocks, settings.chunk_size, settings.chunk_overlap)
    except NoTextLayerError as error:
        _fail(document_id, str(error))
        return 0
    except Exception as error:  # noqa: BLE001 — любая ошибка разбора превращается в отказ документа, а не в падение процесса
        _fail(document_id, f"ошибка разбора: {error!r}")
        return 0

    with connection() as conn:
        saved = chunks_repo.save_chunks(conn, document_id, [vars(chunk) for chunk in chunks])
        documents_repo.set_status(conn, document_id, "indexed")
    log.info("document_indexed", document_id=document_id, chunks=saved)
    _embed_chunks(document_id, chunks)
    return saved


def _embed_chunks(document_id: str, chunks: list[Chunk]) -> None:
    """Посчитать и сохранить эмбеддинги фрагментов разобранного документа.

    Отдельный шаг после сохранения: идентификаторы читаются в порядке
    seq, он совпадает с порядком текстов — векторы встают попарно.
    Ошибка эмбеддинга ловится здесь и уходит в лог: фрагменты остаются
    без вектора, их доберёт команда досчёта.
    """
    try:
        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed_documents(texts)
        with connection() as conn:
            chunk_ids = chunks_repo.list_chunk_ids(conn, document_id)
            chunks_repo.save_embeddings(conn, chunk_ids, vectors)
        log.info("document_embedded", document_id=document_id, vectors=len(vectors))
    except Exception as error:  # noqa: BLE001 — эмбеддинг не должен ронять разбор
        log.warning("document_embed_failed", document_id=document_id, error=repr(error))


def _fail(document_id: str, reason: str) -> None:
    """Поставить документу отказ с причиной."""
    with connection() as conn:
        documents_repo.set_status(conn, document_id, "failed", error=reason)
    log.warning("document_failed", document_id=document_id, reason=reason)
