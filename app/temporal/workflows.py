"""CardWorkflow: extract → (generate → critique)* → сигнал оператора → финал.

Правила детерминизма workflow-кода:
- никаких datetime.now / random / HTTP / чтения конфига — только логика;
- все side-effects — в activities (app/temporal/activities.py);
- таймауты и лимиты — константы модуля, не settings (конфиг меняется
  между реплеями и ломает воспроизведение истории).

Статус задачи пишется в БД отдельным шагом процесса (set_status_activity),
а не изнутри стадий пайплайна. Повтор шага настроен явно (ACTIVITY_RETRY):
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
    from app.schemas.cards import CritiqueReport
    from app.temporal.activities import (
        critique_activity,
        extract_activity,
        generate_activity,
        set_status_activity,
    )

ACTIVITY_TIMEOUT = timedelta(seconds=300)
STATUS_TIMEOUT = timedelta(seconds=30)
ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=3,
)


@workflow.defn
class CardWorkflow:
    """Устойчивый процесс генерации карточки с ожиданием решения оператора."""

    def __init__(self) -> None:
        self._status = "starting"
        self._human_decision: str | None = None

    @workflow.run
    async def run(self, supplier_text: str, job_id: str, max_attempts: int = 3) -> str:
        """Прогнать пайплайн и дождаться решения оператора.

        Статусная машина (зеркалит таблицу jobs):
        extracting → generating → critiquing → (regenerating)*
        → awaiting_confirmation → approved | rejected; при сбое — failed.

        Возвращает JSON итогового черновика карточки.
        """
        try:
            self._status = "extracting"
            await self._set_status(job_id, "extracting")
            facts_json = await workflow.execute_activity(
                extract_activity,
                supplier_text,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=ACTIVITY_RETRY,
            )

            feedback: list[str] = []
            attempts = 0
            draft_json = "{}"
            while attempts < max_attempts:
                attempts += 1
                self._status = f"generating (attempt {attempts})"
                await self._set_status(job_id, "generating", bump_attempts=True)
                draft_json = await workflow.execute_activity(
                    generate_activity,
                    args=[
                        facts_json,
                        json.dumps(feedback, ensure_ascii=False) if feedback else None,
                    ],
                    start_to_close_timeout=ACTIVITY_TIMEOUT,
                    retry_policy=ACTIVITY_RETRY,
                )
                self._status = f"critiquing (attempt {attempts})"
                await self._set_status(job_id, "critiquing")
                report = CritiqueReport.model_validate_json(
                    await workflow.execute_activity(
                        critique_activity,
                        args=[facts_json, draft_json],
                        start_to_close_timeout=ACTIVITY_TIMEOUT,
                        retry_policy=ACTIVITY_RETRY,
                    )
                )
                if report.verdict == "approve":
                    break
                feedback = report.issues
                if attempts < max_attempts:
                    await self._set_status(job_id, "regenerating")

            self._status = "awaiting human decision"
            await self._set_status(job_id, "awaiting_confirmation")
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
        except Exception as exc:
            await self._set_status(job_id, "failed", error=repr(exc))
            raise

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
