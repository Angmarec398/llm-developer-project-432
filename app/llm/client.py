"""LLM-клиент как слой: обёртка над OpenAI Agents SDK с таймаутом,
ретраями и учётом вызовов.

Единственное место в приложении, которое знает про LLM API:
retry-политика, таймаут, различие ошибок 429/5xx/4xx — всё здесь.
Сервисы вызывают run_agent() и не знают про провайдера.

Учёт: после каждого успешного вызова клиент САМ пишет строку
в таблицу llm_calls — задача из контекста, токены из ответа, длительность,
стоимость из цен настроек. Вызывающий код учесть вызов не может забыть:
все вызовы идут через одну границу.
"""
import asyncio
import random
import time
from decimal import Decimal

import httpx
from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.core.context import current_job_id, current_request_id
from app.core.logging import get_logger
from app.llm.cost import call_cost
from app.repositories import llm_calls as llm_calls_repo

log = get_logger(__name__)

_client_configured = False


def setup_llm() -> None:
    """Инициализация OpenAI-compatible клиента (LM Studio / OpenRouter).
    Вызывается один раз на процесс. Идемпотентна.
    """
    global _client_configured
    if _client_configured:
        return
    # keep-alive отключён: LM Studio рвёт простаивающие соединения (RST),
    # и ProactorEventLoop на Windows шумит ConnectionResetError в логах
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=0),
        timeout=settings.llm_timeout_s,
    )
    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_s,
        http_client=http_client,
    )
    set_default_openai_client(client)
    # LM Studio не знает Responses API
    set_default_openai_api("chat_completions")
    # иначе 401 на не-OpenAI ключе
    set_tracing_disabled(True)
    _client_configured = True


def _retryable(exc: Exception) -> bool:
    """Ретраим только транспортные и серверные ошибки: 429, 5xx, таймаут, сеть.

    4xx (кроме 429) — ошибка запроса, повтор бессмыслен.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


async def run_agent(agent: Agent, prompt: str):
    """Запустить агента с ретраями: экспоненциальная задержка + jitter.
    Ретраится только вызов LLM — не вся бизнес-задача (это работа очереди/workflow).
    После успеха вызов записывается в таблицу учёта (_record_call).
    """
    setup_llm()
    max_retries = settings.llm_max_retries
    for attempt in range(1, max_retries + 1):
        try:
            started_at = time.monotonic()
            result = await asyncio.wait_for(
                Runner.run(agent, prompt),
                timeout=settings.llm_timeout_s,
            )
            _record_call(agent, result, started_at)
            return result.final_output
        except Exception as exc:
            if attempt == max_retries or not _retryable(exc):
                log.error(
                    "llm_call_failed",
                    agent=agent.name,
                    attempt=attempt,
                    error=repr(exc),
                )
                raise
            delay = min(2**attempt * 0.5, 8.0) * (1 + random.random() * 0.25)
            log.warning(
                "llm_call_retry",
                agent=agent.name,
                attempt=attempt,
                delay_s=round(delay, 2),
                error=repr(exc),
            )
            await asyncio.sleep(delay)


def _record_call(agent: Agent, result, started_at: float) -> None:
    """Записать вызов в таблицу учёта: токены из ответа, длительность, id из контекста.

    Токены суммируются по raw_responses — на каждый запрос модели своя
    строка расхода. Длительность — всего run_agent включая ретраи.
    Сбой записи не роняет вызов: учёт — наблюдение, а не бизнес-логика.
    """
    duration_ms = int((time.monotonic() - started_at) * 1000)
    prompt_tokens = 0
    completion_tokens = 0
    for response in getattr(result, "raw_responses", None) or []:
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens += usage.input_tokens or 0
            completion_tokens += usage.output_tokens or 0
    model = str(getattr(agent, "model", None) or settings.llm_model)
    price_input, price_output = _prices_for(model)
    cost = call_cost(
        prompt_tokens, completion_tokens, price_input, price_output
    )
    try:
        llm_calls_repo.record_call(
            job_id=current_job_id(),
            request_id=current_request_id(),
            agent=agent.name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            duration_ms=duration_ms,
        )
    except Exception as error:  # noqa: BLE001 — учёт не должен ронять вызов
        log.warning("llm_call_record_failed",
                    agent=agent.name, error=repr(error))


def _prices_for(model: str) -> tuple[Decimal, Decimal]:
    """Цены за миллион токенов для модели: служебная — дешёвые, остальные — основные.

    Политика двух моделей: критик, судья и починка полей идут
    на llm_model_utility с её ценами; генерация карточки — на основную.
    """
    if model == settings.llm_model_utility:
        return (
            settings.llm_utility_price_input_per_mtok,
            settings.llm_utility_price_output_per_mtok,
        )
    return (
        settings.llm_price_input_per_mtok,
        settings.llm_price_output_per_mtok,
    )
