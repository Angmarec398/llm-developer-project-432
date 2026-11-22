"""Команда досчёта эмбеддингов: фрагментам без вектора — векторы.

Пригодится и после сбоя индексации, и после смены модели. Идемпотентна:
повторный запуск ничего не пересчитывает — нечему, векторы уже есть.
Прогресс пишется в лог пачками.

    python -m app.rag.reindex
"""
from app.core.db import connection
from app.core.logging import get_logger, setup_logging
from app.rag import embedder
from app.repositories import chunks as chunks_repo

log = get_logger(__name__)

BATCH_SIZE = 64  # сколько фрагментов досчитывается за одну итерацию


def reindex(batch_size: int = BATCH_SIZE) -> int:
    """Досчитать векторы всем фрагментам без них, вернуть число записанных.

    Каждая итерация: выбрать пачку фрагментов без вектора, закодировать
    их одним вызовом, записать пачкой. Ошибка пачки обрывает прогон —
    оставшиеся фрагменты попадут в следующий запуск, их никто не потерял.
    """
    setup_logging()
    total_missing = _count_missing()
    if total_missing == 0:
        log.info("reindex_nothing_to_do")
        return 0
    log.info("reindex_started", missing=total_missing)
    done = 0
    while True:
        with connection() as conn:
            batch = chunks_repo.missing_embeddings(conn, limit=batch_size)
        if not batch:
            break
        chunk_ids = [chunk_id for chunk_id, _text in batch]
        texts = [text for _chunk_id, text in batch]
        try:
            vectors = embedder.embed_documents(texts)
        except Exception as error:  # не обрывать досчёт серией ошибок
            log.error("reindex_batch_failed", first_chunk_id=batch[0][0], error=repr(error))
            break
        with connection() as conn:
            saved = chunks_repo.save_embeddings(conn, chunk_ids, vectors)
        done += saved
        log.info("reindex_progress", done=done, total=total_missing)
    log.info("reindex_finished", embedded=done)
    return done


def _count_missing() -> int:
    """Сколько фрагментов сейчас без вектора."""
    with connection() as conn:
        return chunks_repo.count_missing_embeddings(conn)


def main() -> None:
    """Точка входа: досчитать векторы и напечатать итог."""
    embedded = reindex()
    print(f"досчитано векторов: {embedded}")


if __name__ == "__main__":
    main()
