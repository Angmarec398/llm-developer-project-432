"""Общая точка доступа к Temporal-серверу.

Клиент — ресурс уровня процесса: создаётся один раз при первом
обращении и переиспользуется роутерами и воркером.
"""
from temporalio.client import Client

from app.core.config import settings

_client: Client | None = None


async def get_client() -> Client:
    """Вернуть подключённый Temporal-клиент, создав его при первом вызове."""
    global _client
    if _client is None:
        _client = await Client.connect(settings.temporal_host)
    return _client
