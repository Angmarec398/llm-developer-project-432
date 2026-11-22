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
from app.guardrails import injection
from app.guardrails.output import filter_card_output
from app.guardrails.pii import mask_pii, masked_labels
from app.rag import retrieval
from app.rag.context import build_context
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas.cards import (
    ExcludedChunk,
    GuardrailReport,
    MaskedItem,
    ProductCard,
    QuarantinedDocument,
)
from app.services import citations, pipeline
from app.services.documents import process_document

log = get_logger(__name__)

_EMBED_BATCH = 64  # сколько фрагментов доиндексируется за одну пачку


class GuardrailBlockedError(ValueError):
    """Все фрагменты отклонены защитой: генерировать не из чего (шаг 9).

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
    разобранным. Уже проиндексированные не трогает — повторный разбор
    дал бы дубли фрагментов.
    """
    parsed = {}
    with connection() as conn:
        documents = {
            document_id: documents_repo.get_document(conn, document_id)
            for document_id in document_ids
        }
    for document_id, document in documents.items():
        if document is None or document["status"] == "indexed":
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
    from app.rag import embedder  # тяжёлая модель — импорт по месту вызова

    embedded = 0
    with connection() as conn:
        missing = chunks_repo.missing_embeddings_for_documents(
            conn, document_ids, limit=_EMBED_BATCH
        )
    while missing:
        chunk_ids = [chunk_id for chunk_id, _text in missing]
        texts = [text for _chunk_id, text in missing]
        try:
            vectors = embedder.embed_documents(texts)
            with connection() as conn:
                embedded += chunks_repo.save_embeddings(conn, chunk_ids, vectors)
        except Exception as error:  # noqa: BLE001 — эмбеддинг не должен ронять задачу
            log.warning("embed_batch_failed", error=repr(error))
            break
        with connection() as conn:
            missing = chunks_repo.missing_embeddings_for_documents(
                conn, document_ids, limit=_EMBED_BATCH
            )
    return embedded


def search_chunks(
    document_ids: list[str], hint: str | None
) -> tuple[list[dict], GuardrailReport]:
    """Найти фрагменты и замаскировать в них ПДн (первый рубеж, шаг 9).

    Запрос — подсказка оператора, а без неё — заголовки (имена файлов)
    выбранных документов. Поиск гибридный, фильтр — по списку выбранных
    документов: модель видит только их фрагменты.

    Маскирование идёт ДО сборки контекста: замаскированный текст уходит
    и в промпт модели, и наружу из activity (в историю процесса
    Temporal) — сырые контакты в чужое API и в логи не попадают.
    Второе возвращаемое значение — отчёт: что замаскировано.
    """
    query = hint or _documents_titles(document_ids)
    results = retrieval.search(
        query, mode="hybrid", filters={"document_id": document_ids}
    )
    report = GuardrailReport()
    masked_results: list[dict] = []
    for chunk in results:
        masked = mask_pii(chunk["text"])
        report.masked.extend(
            MaskedItem(category=category, label=label)
            for category, label in masked_labels(masked.findings)
        )
        masked_results.append({**chunk, "text": masked.text})
    return masked_results, report


def assemble_context(results: list[dict]) -> tuple[str, list[int]]:
    """Собрать контекст из (замаскированных) фрагментов: текст + id."""
    return build_context(results)


async def screen_chunks(
    chunks: list[dict], report: GuardrailReport
) -> tuple[list[dict], GuardrailReport, str | None]:
    """Скрининг фрагментов детектором инъекций (второй рубеж, шаг 9).

    Каждый фрагмент проверяет двухуровневый детектор (правила, затем —
    по помеченным — служебная модель; app/guardrails/injection.py).
    Подозрительные фрагменты исключаются из контекста и попадают в
    отчёт с причинами. Доля подозрительных в документе выше
    injection_document_threshold — документ целиком к человеку: все
    его фрагменты исключаются, в отчёт пишется причина. Третье
    возвращаемое значение — причина блокировки, если чистых
    фрагментов не осталось: генерировать не из чего.
    """
    excluded_ids: set[int] = set()
    total_by_document: dict[str, int] = {}
    suspicious_by_document: dict[str, int] = {}
    for chunk in chunks:
        document_id = chunk["document_id"]
        total_by_document[document_id] = (
            total_by_document.get(document_id, 0) + 1
        )
        screen_result = await injection.screen_text(chunk["text"])
        if not screen_result.suspicious:
            continue
        suspicious_by_document[document_id] = (
            suspicious_by_document.get(document_id, 0) + 1
        )
        report.excluded_chunks.append(
            ExcludedChunk(
                chunk_id=chunk["id"],
                document_id=document_id,
                reasons=screen_result.reasons,
            )
        )
        excluded_ids.add(chunk["id"])
        log.info(
            "guardrail_chunk_excluded",
            chunk_id=chunk["id"],
            reasons=screen_result.reasons,
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
                QuarantinedDocument(document_id=document_id, reason=reason)
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
        if chunk["id"] not in excluded_ids
        and chunk["document_id"] not in quarantined_ids
    ]
    blocked_reason = None
    if not clean:
        blocked_reason = "; ".join(
            f"документ {item.document_id}: {item.reason}"
            for item in report.quarantined_documents
        ) or "все фрагменты отклонены детектором инъекций"
    return clean, report, blocked_reason


def verify_card_citations(
    card: ProductCard, context_chunk_ids: list[int]
) -> list[str]:
    """Проверить ссылки карточки: фрагмент существует и был в контексте."""
    with connection() as conn:
        return citations.verify_citations(conn, card, context_chunk_ids)


async def run_card_pipeline(
    document_ids: list[str],
    hint: str | None = None,
    max_attempts: int = 3,
) -> tuple[ProductCard, int, str, GuardrailReport]:
    """Полный цикл синхронно (POST /cards): подготовка → поиск → генерация.

    Возвращает (карточку, число генераций, вердикт, отчёт защиты).
    Вердикты: approved — критик доволен, ссылки в порядке, уверенность
    выше порога; low_confidence — то же, но уверенность ниже порога:
    карточку должен посмотреть человек; citation_fraud — модель
    повторно ссылается на фрагмент вне контекста (выдумывание с
    правдоподобным видом) — эскалация к человеку; rejected — петля
    генераций исчерпана; output_contamination — в карточке найдены
    следы служебных инструкций: эскалация к человеку. Контекст
    собирается из замаскированных фрагментов (шаг 9): ПДн заменены
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
    context, context_chunk_ids = assemble_context(screened)

    feedback: list[str] = []
    citation_failed_once = False
    attempts = 0
    card: ProductCard | None = None
    approved = False
    while attempts < max_attempts:
        attempts += 1
        card = await pipeline.generate_card(context, feedback or None)
        violations = verify_card_citations(card, context_chunk_ids)
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
        report = await pipeline.critique(context, card)
        if report.verdict == "approve":
            approved = True
            break
        feedback = report.issues
        log.info("pipeline_regenerate", attempt=attempts, issues=report.issues)

    assert card is not None  # цикл исполняется хотя бы раз: max_attempts >= 1
    card, output_traces = filter_card_output(card, guardrail_report)
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
