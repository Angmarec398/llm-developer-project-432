"""Точка входа Temporal-воркера: регистрирует workflows + activities."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from temporalio.worker import Worker

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.temporal.activities import (
    critique_activity,
    embed_documents_activity,
    filter_output_activity,
    generate_activity,
    parse_document_activity,
    parse_documents_activity,
    screen_context_activity,
    search_chunks_activity,
    set_status_activity,
    verify_citations_activity,
)
from app.temporal.client import get_client
from app.temporal.workflows import CardWorkflow, DocumentWorkflow

log = get_logger(__name__)

# Сколько воркер ждёт завершения уже выполняющихся активностей после
# сигнала остановки, прежде чем отменить их. Пока идёт ожидание, воркер
# не берёт новых задач. Значение закрывает самый долгий шаг: LLM-вызов
# с повторами (llm_timeout_s=300) внутри активностей с лимитом 600 с.
GRACEFUL_SHUTDOWN_TIMEOUT = timedelta(minutes=5)


async def main():
    """Поднять воркер на очереди из конфига и обслуживать оба процесса."""
    setup_logging()
    client = await get_client()
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[CardWorkflow, DocumentWorkflow],
        activities=[
            parse_documents_activity,
            embed_documents_activity,
            search_chunks_activity,
            screen_context_activity,
            generate_activity,
            verify_citations_activity,
            critique_activity,
            filter_output_activity,
            set_status_activity,
            parse_document_activity,
        ],
        activity_executor=ThreadPoolExecutor(max_workers=4),
        graceful_shutdown_timeout=GRACEFUL_SHUTDOWN_TIMEOUT,
    )
    log.info("temporal_worker_started", task_queue=settings.temporal_task_queue)
    try:
        await worker.run()
    except asyncio.CancelledError:
        # Ctrl+C: worker.run() уже дождался остановки поллеров и
        # выполнившихся активностей — логируем чистый выход и гасим
        # CancelledError, чтобы asyncio.run не печатал traceback.
        log.info("temporal_worker_stopped")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Остановка по Ctrl+C — штатная операция, не ошибка.
        pass
