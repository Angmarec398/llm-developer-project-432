"""CardWorkflow: документы → поиск → генерация → проверка → человек.

DocumentWorkflow: разбор загруженного документа в фрагменты (шаг 5).

Правила детерминизма workflow-кода:
- никаких datetime.now / random / HTTP / чтения конфига — только логика;
- все side-effects — в activities (app/temporal/activities.py);
- таймауты и лимиты — константы модуля или аргументы workflow, не settings
  (конфиг меняется между реплеями и ломает воспроизведение истории).

Статус задачи пишется в БД отдельным шагом процесса (set_status_activity),
а не изнутри стадий. Повтор шага настроен явно (ACTIVITY_RETRY):
упавший шаг повторяется движком, убитый воркер подхватывает процесс
с последнего завершённого шага.

Песочница воркфлоу запрещает тяжёлые импорты, поэтому app.schemas
и activities проходят через workflow.unsafe.imports_passed_through().
"""
import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.schemas.cards import CritiqueReport, ProductCard
    from app.temporal.activities import (
        critique_activity,
        embed_documents_activity,
        generate_activity,
        parse_document_activity,
        parse_documents_activity,
        search_context_activity,
        set_status_activity,
        verify_citations_activity,
    )

ACTIVITY_TIMEOUT = timedelta(seconds=300)
STATUS_TIMEOUT = timedelta(seconds=30)
ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=3,
)


@workflow.defn
class DocumentWorkflow:
    """Устойчивый процесс разбора документа: один блокирующий шаг.

    Статусы документа (parsing → indexed | failed) пишет сам сервис
    разбора внутри шага — документ не задача генерации, отдельная
    status-activity ему не нужна. workflow_id детерминированный:
    document-<document_id>, повторный старт с тем же id движок
    не выполнит.
    """

    @workflow.run
    async def run(self, document_id: str, file_path: str) -> int:
        """Запустить шаг разбора и вернуть число сохранённых фрагментов."""
        return await workflow.execute_activity(
            parse_document_activity,
            args=[document_id, file_path],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=ACTIVITY_RETRY,
        )


@workflow.defn
class CardWorkflow:
    """Устойчивый процесс генерации карточки от контекста (шаг 7).

    Статусная машина (зеркалит таблицу jobs):
    parsing → indexing → searching → generating → verifying → critiquing
    → (regenerating)* → done | awaiting_confirmation → approved | rejected;
    при сбое — failed.
    """

    def __init__(self) -> None:
        self._status = "starting"
        self._human_decision: str | None = None

    @workflow.run
    async def run(
        self,
        job_id: str,
        document_ids: list[str],
        hint: str | None = None,
        max_attempts: int = 3,
        confidence_threshold: float = 0.6,
    ) -> str:
        """Прогнать сквозной сценарий и вернуть JSON итоговой карточки.

        Три рубежа против выдумывания: модель видит только найденное
        (поиск по выбранным документам), каждая ссылка проверяется
        (фрагмент есть в базе и был в поданном контексте), критик
        сверяет утверждения с источниками. Выдуманная ссылка — переделка
        с текстом ошибки; повторное нарушение — ожидание человека.
        Не одобрено критиком или уверенность ниже порога — тоже
        ожидание человека. Порог приходит аргументом (из настроек
        приложения): workflow не читает конфиг — значение фиксируется
        в истории.
        """
        try:
            self._status = "parsing"
            await self._set_status(job_id, "parsing")
            parse_result = json.loads(
                await workflow.execute_activity(
                    parse_documents_activity,
                    json.dumps(document_ids),
                    start_to_close_timeout=ACTIVITY_TIMEOUT,
                    retry_policy=ACTIVITY_RETRY,
                )
            )
            if parse_result["problems"]:
                problems = "; ".join(parse_result["problems"])
                self._status = "failed"
                await self._set_status(job_id, "failed", error=problems)
                return "{}"

            self._status = "indexing"
            await self._set_status(job_id, "indexing")
            await workflow.execute_activity(
                embed_documents_activity,
                json.dumps(document_ids),
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=ACTIVITY_RETRY,
            )

            self._status = "searching"
            await self._set_status(job_id, "searching")
            bundle = json.loads(
                await workflow.execute_activity(
                    search_context_activity,
                    args=[json.dumps(document_ids), hint],
                    start_to_close_timeout=ACTIVITY_TIMEOUT,
                    retry_policy=ACTIVITY_RETRY,
                )
            )
            context = bundle["context"]
            context_chunk_ids = bundle["chunk_ids"]

            feedback: list[str] = []
            attempts = 0
            draft_json = "{}"
            approved = False
            citation_failed_once = False
            while attempts < max_attempts:
                attempts += 1
                self._status = f"generating (attempt {attempts})"
                await self._set_status(job_id, "generating", bump_attempts=True)
                draft_json = await workflow.execute_activity(
                    generate_activity,
                    args=[
                        context,
                        json.dumps(feedback, ensure_ascii=False)
                        if feedback else None,
                    ],
                    start_to_close_timeout=ACTIVITY_TIMEOUT,
                    retry_policy=ACTIVITY_RETRY,
                )

                self._status = "verifying citations"
                await self._set_status(job_id, "verifying")
                violations = json.loads(
                    await workflow.execute_activity(
                        verify_citations_activity,
                        args=[draft_json, json.dumps(context_chunk_ids)],
                        start_to_close_timeout=ACTIVITY_TIMEOUT,
                        retry_policy=ACTIVITY_RETRY,
                    )
                )
                if violations:
                    if citation_failed_once:
                        return await self._await_human(
                            job_id, draft_json,
                            "выдуманные ссылки: " + "; ".join(violations),
                        )
                    citation_failed_once = True
                    feedback = ["Исправь ссылки на источники:"] + [
                        f"- {violation}" for violation in violations
                    ]
                    if attempts < max_attempts:
                        await self._set_status(job_id, "regenerating")
                    continue

                self._status = f"critiquing (attempt {attempts})"
                await self._set_status(job_id, "critiquing")
                report = CritiqueReport.model_validate_json(
                    await workflow.execute_activity(
                        critique_activity,
                        args=[context, draft_json],
                        start_to_close_timeout=ACTIVITY_TIMEOUT,
                        retry_policy=ACTIVITY_RETRY,
                    )
                )
                if report.verdict == "approve":
                    approved = True
                    break
                feedback = report.issues
                if attempts < max_attempts:
                    await self._set_status(job_id, "regenerating")

            card = ProductCard.model_validate_json(draft_json)
            if approved and card.confidence >= confidence_threshold:
                self._status = "done"
                await self._set_status(job_id, "done", result_json=draft_json)
                return draft_json

            reason = (
                "уверенность ниже порога"
                if approved
                else "критик не одобрил карточку"
            )
            return await self._await_human(job_id, draft_json, reason)
        except Exception as exc:
            await self._set_status(job_id, "failed", error=repr(exc))
            raise

    async def _await_human(
        self, job_id: str, draft_json: str, reason: str
    ) -> str:
        """Устойчивая пауза: карточка ждёт сигнала оператора.

        Задача живёт в базе в статусе awaiting_confirmation и переживает
        что угодно; человек смотрит карточку со ссылками на источники
        и решает сигналом approve или reject.
        """
        self._status = f"awaiting human decision ({reason})"
        await self._set_status(
            job_id, "awaiting_confirmation",
            result_json=draft_json, error=reason,
        )
        await workflow.wait_condition(lambda: self._human_decision is not None)

        if self._human_decision == "approve":
            self._status = "completed"
            await self._set_status(job_id, "approved", result_json=draft_json)
        else:
            self._status = "rejected by human"
            await self._set_status(
                job_id, "rejected",
                result_json=draft_json, error="rejected by operator",
            )
        return draft_json

    async def _set_status(
        self,
        job_id: str,
        status: str,
        *,
        result_json: str | None = None,
        error: str | None = None,
        bump_attempts: bool = False,
    ) -> None:
        """Отдельный шаг процесса: записать статус задачи в БД."""
        await workflow.execute_activity(
            set_status_activity,
            args=[job_id, status, result_json, error, bump_attempts],
            start_to_close_timeout=STATUS_TIMEOUT,
            retry_policy=ACTIVITY_RETRY,
        )

    @workflow.signal
    def approve(self) -> None:
        """Сигнал оператора: карточка подтверждена."""
        self._human_decision = "approve"

    @workflow.signal
    def request_changes(self) -> None:
        """Сигнал оператора: карточка отклонена, нужны правки."""
        self._human_decision = "regenerate"

    @workflow.query
    def current_status(self) -> str:
        """Запрос текущего состояния процесса (из памяти workflow)."""
        return self._status
