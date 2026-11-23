"""Оркестрация карточки: документы → поиск → защита → генерация → проверки.

Сквозной сценарий: на входе список документов и необязательная
подсказка, что это за товар; дальше цепочка разбор → индексация →
поиск → маскирование ПДн → скрининг инъекций → сборка контекста →
генерация → проверка ссылок → критик → фильтр выхода; на выходе либо
согласовано, либо карточка ждёт человека.

Функции мелкие и самостоятельные: Temporal-activity каждого шага
вызывает ровно одну из них (app/temporal/activities.py), а
run_card_pipeline собирает их в синхронный цикл для POST /cards.
"""
from app.core.config import settings
from app.core.db import connection
from app.core.logging import get_logger
from app.guardrails import guard_output
from app.guardrails.injection import detect_injection_llm, detect_injection_regex
from app.guardrails.pii import mask_pii
from app.rag import retrieval
from app.rag.context import build_context, context_chunk_ids
from app.rag.reindex import index_pending_chunks
from app.repositories import documents as documents_repo
from app.schemas.cards import (
    CardDraft,
    ExcludedChunk,
    GuardrailReport,
    MaskedItem,
    QuarantinedDocument,
)
from app.services import citations
from app.services.documents import process_document
from app.services.rag_pipeline import _coerce_card, critique_with_context

log = get_logger(__name__)

_EMBED_BATCH = 64  # сколько фрагментов доиндексируется за одну пачку


class GuardrailBlockedError(ValueError):
    """Все фрагменты отклонены защитой: генерировать не из чего.

    Синхронный эндпоинт (POST /cards) превращает её в ответ 422 с
    причиной; устойчивый процесс (CardWorkflow) вместо исключения
    получает blocked_reason из screen_context_activity и уходит в
    awaiting_confirmation. Отчёт защиты — в атрибуте report.
    """

    def __init__(
        self, reason: str, report: GuardrailReport | None = None
    ) -> None:
        """Запомнить причину блокировки и отчёт защиты."""
        super().__init__(reason)
        self.report = report


def documents_problems(document_ids: list[str]) -> list[str]:
    """Проверить, что все документы на месте и не в отказе.

    Возвращает список проблем (пустой — документы готовы): документ
    не найден или разобран с ошибкой — с причиной из базы.
    """
    problems = []
    with connection() as conn:
        for document_id in document_ids:
            document = documents_repo.get_document(conn, document_id)
            if document is None:
                problems.append(f"документ {document_id} не найден")
            elif document["status"] == "failed":
                problems.append(
                    f"документ {document_id} не разобран: {document['error']}"
                )
    return problems


def parse_pending_documents(document_ids: list[str]) -> dict[str, int]:
    """Разобрать документы, которые ещё не проиндексированы.

    Возвращает {document_id: число сохранённых фрагментов} по реально
    разобранным. Уже проиндексированные не трогаем — повторный разбор
    дал бы дубли фрагментов.
    """
    parsed = {}
    with connection() as conn:
        documents = {
            document_id: documents_repo.get_document(conn, document_id)
            for document_id in document_ids
        }
    for document_id, document in documents.items():
        if document is None or document["status"] in ("indexed", "embed_failed"):
            continue
        saved = process_document(document_id, document["file_path"])
        if saved:
            parsed[document_id] = saved
    return parsed


def embed_pending_documents(document_ids: list[str]) -> int:
    """Досчитать эмбеддинги фрагментам выбранных документов, вернуть число.

    Идемпотентно: вектор ставится только тем, у кого его нет. Ошибка
    пачки ловится и идёт в лог — без вектора фрагменты доберёт
    команда досчёта (make reindex), задача из-за этого не падает.
    """
    embedded = 0
    for document_id in document_ids:
        try:
            with connection() as conn:
                embedded += index_pending_chunks(
                    conn, doc_id=document_id, batch=_EMBED_BATCH
                )
        except Exception as error:  # noqa: BLE001 — эмбеддинг не должен ронять задачу
            log.warning("embed_batch_failed", document_id=document_id, error=repr(error))
    return embedded


def search_chunks(
    document_ids: list[str], hint: str | None
) -> tuple[list[dict], GuardrailReport]:
    """Найти фрагменты и замаскировать в них ПДн (первый рубеж защиты).

    Запрос — подсказка оператора, а без неё — заголовки (имена файлов)
    выбранных документов. Поиск гибридный, фильтр — по списку выбранных
    документов: модель видит только их фрагменты.

    Маскирование идёт ДО сборки контекста: замаскированный текст уходит
    и в промпт модели, и наружу из activity (в историю процесса
    Temporal) — сырые контакты в чужое API и в логи не попадают.
    Второе возвращаемое значение — отчёт: что замаскировано.
    """
    query = hint or _documents_titles(document_ids)
    results = retrieval.search(query, mode="hybrid", doc_ids=document_ids)
    report = GuardrailReport()
    masked_results: list[dict] = []
    for chunk in results:
        masked_text, entities = mask_pii(chunk["text"])
        report.masked.extend(
            MaskedItem(category=entity.kind, label=entity.placeholder)
            for entity in entities
        )
        masked_results.append({**chunk, "text": masked_text})
    return masked_results, report


def assemble_context(results: list[dict]) -> tuple[str, list[str]]:
    """Собрать контекст из (замаскированных) фрагментов: текст + id вошедших."""
    context = build_context(results)
    return context, sorted(context_chunk_ids(context))


async def screen_chunks(
    chunks: list[dict], report: GuardrailReport
) -> tuple[list[dict], GuardrailReport, str | None]:
    """Скрининг фрагментов детектором инъекций (второй рубеж защиты).

    Каждый фрагмент проверяет двухуровневый детектор (правила, затем —
    по помеченным — служебная модель; app/guardrails/injection.py).
    Подозрительные фрагменты исключаются из контекста и попадают в
    отчёт с причинами. Доля подозрительных в документе выше
    injection_document_threshold — документ целиком к человеку: все
    его фрагменты исключаются, в отчёт пишется причина. Третье
    возвращаемое значение — причина блокировки, если чистых
    фрагментов не осталось: генерировать не из чего.
    """
    if not chunks:
        return [], report, (
            "поиск не вернул ни одного фрагмента по выбранным документам"
        )
    excluded_ids: set[str] = set()
    total_by_document: dict[str, int] = {}
    suspicious_by_document: dict[str, int] = {}
    for chunk in chunks:
        document_id = chunk["doc_id"]
        total_by_document[document_id] = (
            total_by_document.get(document_id, 0) + 1
        )
        regex_verdict = detect_injection_regex(chunk["text"])
        if not regex_verdict.suspicious:
            continue
        reasons = [regex_verdict.reason]
        try:
            llm_verdict = await detect_injection_llm(chunk["text"])
            confirmed = llm_verdict.suspicious
            if llm_verdict.reason:
                reasons.append(llm_verdict.reason)
        except Exception as error:  # noqa: BLE001 — fail-closed: сбой детектора = фрагмент подозрительный
            log.warning(
                "injection_llm_failed",
                chunk_id=chunk["chunk_id"], error=repr(error),
            )
            confirmed = True
        if not confirmed:
            continue
        suspicious_by_document[document_id] = (
            suspicious_by_document.get(document_id, 0) + 1
        )
        report.excluded_chunks.append(
            ExcludedChunk(
                chunk_id=chunk["chunk_id"],
                doc_id=document_id,
                reasons=reasons,
            )
        )
        excluded_ids.add(chunk["chunk_id"])
        log.info(
            "guardrail_chunk_excluded",
            chunk_id=chunk["chunk_id"],
            reasons=reasons,
        )

    quarantined_ids: set[str] = set()
    for document_id, total in total_by_document.items():
        suspicious = suspicious_by_document.get(document_id, 0)
        share = suspicious / total
        if share > settings.injection_document_threshold:
            reason = (
                f"подозрительных фрагментов {suspicious} из {total} "
                f"(порог {settings.injection_document_threshold})"
            )
            report.quarantined_documents.append(
                QuarantinedDocument(doc_id=document_id, reason=reason)
            )
            quarantined_ids.add(document_id)
            log.warning(
                "guardrail_document_quarantined",
                document_id=document_id,
                reason=reason,
            )

    clean = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] not in excluded_ids
        and chunk["doc_id"] not in quarantined_ids
    ]
    blocked_reason = None
    if not clean:
        blocked_reason = "; ".join(
            f"документ {item.doc_id}: {item.reason}"
            for item in report.quarantined_documents
        ) or "все фрагменты отклонены детектором инъекций"
    return clean, report, blocked_reason


async def generate_card(context: str, feedback: list[str] | None = None) -> CardDraft:
    """Сгенерировать карточку по контексту (без ретраев — ими управляет цикл).

    Ссылки на источники проверяет отдельный шаг (verify_card_citations):
    генератор и проверка разнесены, чтобы нарушение было видно в истории
    процесса своим состоянием.
    """
    from app.agents.prompts import context_generator_agent
    from app.llm.client import run_agent
    from app.rag.context import CONTEXT_INSTRUCTION

    prefix = ""
    if feedback:
        prefix = "Замечания, учти:\n" + "\n".join(f"- {issue}" for issue in feedback) + "\n\n"
    out = await run_agent(
        context_generator_agent, prefix + CONTEXT_INSTRUCTION + "\nКОНТЕКСТ:\n" + context
    )
    return _coerce_card(out)


def verify_card_citations(
    card: CardDraft, context_ids: list[str]
) -> list[str]:
    """Проверить ссылки карточки: фрагмент существует и был в контексте."""
    with connection() as conn:
        return citations.verify_citations(conn, card, context_ids)


def filter_card_output(
    card: CardDraft, report: GuardrailReport
) -> tuple[CardDraft, GuardrailReport, list[str]]:
    """Фильтр выхода (третий рубеж): ПДн в карточке маскируются, следы
    служебных инструкций — эскалация к человеку (непустой третий элемент).
    """
    card_json = card.model_dump_json()
    masked_json, entities = mask_pii(card_json)
    if entities:
        report.output_masked.extend(
            MaskedItem(category=entity.kind, label=entity.placeholder)
            for entity in entities
        )
    cleaned_json, issues = guard_output(masked_json)
    traces = [issue for issue in issues if "след инъекции" in issue]
    return CardDraft.model_validate_json(cleaned_json), report, traces


async def run_card_pipeline(
    document_ids: list[str],
    hint: str | None = None,
    max_attempts: int = 3,
) -> tuple[CardDraft, int, str, GuardrailReport]:
    """Полный цикл синхронно (POST /cards): подготовка → поиск → генерация.

    Возвращает (карточку, число генераций, вердикт, отчёт защиты).
    Вердикты: approved — критик доволен, ссылки в порядке, уверенность
    выше порога; low_confidence — то же, но уверенность ниже порога:
    карточку должен посмотреть человек; citation_fraud — модель
    повторно ссылается на фрагмент вне контекста (выдумывание с
    правдоподобным видом) — эскалация к человеку; rejected — петля
    генераций исчерпана; output_contamination — в карточке найдены
    следы служебных инструкций: эскалация к человеку. Контекст
    собирается из замаскированных фрагментов: ПДн заменены
    метками до промпта и до логов. Все фрагменты отклонены защитой —
    GuardrailBlockedError с причиной и отчётом (эндпоинт отвечает 422).
    """
    problems = documents_problems(document_ids)
    if problems:
        raise ValueError("; ".join(problems))
    parse_pending_documents(document_ids)
    embed_pending_documents(document_ids)
    results, guardrail_report = search_chunks(document_ids, hint)
    screened, guardrail_report, blocked_reason = await screen_chunks(
        results, guardrail_report
    )
    if blocked_reason is not None:
        raise GuardrailBlockedError(blocked_reason, guardrail_report)
    context, context_ids = assemble_context(screened)

    feedback: list[str] = []
    citation_failed_once = False
    attempts = 0
    card: CardDraft | None = None
    approved = False
    while attempts < max_attempts:
        attempts += 1
        card = await generate_card(context, feedback or None)
        violations = verify_card_citations(card, context_ids)
        if violations:
            log.info(
                "pipeline_citation_violation",
                attempt=attempts, violations=violations,
            )
            feedback = ["Исправь ссылки на источники:"] + [
                f"- {violation}" for violation in violations
            ]
            if citation_failed_once:
                return card, attempts, "citation_fraud", guardrail_report
            citation_failed_once = True
            continue
        report = await critique_with_context(context, card)
        if report.verdict == "approve":
            approved = True
            break
        feedback = report.issues
        log.info("pipeline_regenerate", attempt=attempts, issues=report.issues)

    assert card is not None  # цикл исполняется хотя бы раз: max_attempts >= 1
    card, guardrail_report, output_traces = filter_card_output(card, guardrail_report)
    if output_traces:
        log.warning("guardrail_output_contamination", traces=output_traces)
        return card, attempts, "output_contamination", guardrail_report
    verdict = "approved" if approved else "rejected"
    if approved and card.confidence < settings.card_confidence_threshold:
        verdict = "low_confidence"
        log.info("pipeline_low_confidence", confidence=card.confidence)
    return card, attempts, verdict, guardrail_report


def _documents_titles(document_ids: list[str]) -> str:
    """Заголовки документов (имена файлов) через точку с запятой."""
    with connection() as conn:
        titles = [
            document["filename"]
            for document_id in document_ids
            if (document := documents_repo.get_document(conn, document_id))
            is not None
        ]
    return "; ".join(titles)
