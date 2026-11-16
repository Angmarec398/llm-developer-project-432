"""Точка входа Temporal-воркера: регистрирует workflows + activities."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.worker import Worker

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.temporal.activities import (
    critique_activity,
    extract_activity,
    generate_activity,
    set_status_activity,
)
from app.temporal.client import get_client
from app.temporal.workflows import CardWorkflow

log = get_logger(__name__)


async def main():
    setup_logging()
    client = await get_client()
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[CardWorkflow],
        activities=[extract_activity, generate_activity, critique_activity, set_status_activity],
        activity_executor=ThreadPoolExecutor(max_workers=4),
    )
    log.info("temporal_worker_started", task_queue=settings.temporal_task_queue)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
