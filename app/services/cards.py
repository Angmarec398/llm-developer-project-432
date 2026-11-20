"""Оркестрация карточки шага 7: документы → поиск → генерация → проверки.

Сквозной сценарий: на входе список документов и необязательная
подсказка, что это за товар; дальше цепочка разбор → индексация →
поиск → генерация → проверка ссылок → критик; на выходе либо
согласовано, либо карточка ждёт человека.

Функции мелкие и самостоятельные: Temporal-activity каждого шага
вызывает ровно одну из них (app/temporal/activities.py), а
run_card_pipeline собирает их в синхронный цикл для POST /cards.
"""
from app.core.config import settings
from app.core.db import connection
from app.core.logging import get_logger
from app.rag import retrieval
from app.rag.context import build_context
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.schemas.cards import ProductCard
from app.services import citations, pipeline
from app.services.documents import process_document

log = get_logger(__name__)

_EMBED_BATCH = 64  # сколько фрагментов доиндексируется за одну пачку


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


def build_search_context(
    document_ids: list[str], hint: str | None
) -> tuple[str, list[int]]:
    """Найти фрагменты и собрать контекст: текст + id вошедших фрагментов.

    Запрос — подсказка оператора, а без неё — заголовки (имена файлов)
    выбранных документов. Поиск гибридный, фильтр — по списку выбранных
    документов: модель видит только их фрагменты.
    """
    query = hint or _documents_titles(document_ids)
    results = retrieval.search(
        query, mode="hybrid", filters={"document_id": document_ids}
    )
    return build_context(results)


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
) -> tuple[ProductCard, int, str]:
    """Полный цикл синхронно (POST /cards): подготовка → поиск → генерация.

    Возвращает (карточку, число генераций, вердикт). Вердикты:
    approved — критик доволен, ссылки в порядке, уверенность выше
    порога; low_confidence — то же, но уверенность ниже порога:
    карточку должен посмотреть человек; citation_fraud — модель
    повторно ссылается на фрагмент вне контекста (выдумывание с
    правдоподобным видом) — эскалация к человеку; rejected — петля
    генераций исчерпана.
    """
    problems = documents_problems(document_ids)
    if problems:
        raise ValueError("; ".join(problems))
    parse_pending_documents(document_ids)
    embed_pending_documents(document_ids)
    context, context_chunk_ids = build_search_context(document_ids, hint)

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
                return card, attempts, "citation_fraud"
            citation_failed_once = True
            continue
        report = await pipeline.critique(context, card)
        if report.verdict == "approve":
            approved = True
            break
        feedback = report.issues
        log.info("pipeline_regenerate", attempt=attempts, issues=report.issues)

    assert card is not None  # цикл исполняется хотя бы раз: max_attempts >= 1
    verdict = "approved" if approved else "rejected"
    if approved and card.confidence < settings.card_confidence_threshold:
        verdict = "low_confidence"
        log.info("pipeline_low_confidence", confidence=card.confidence)
    return card, attempts, verdict


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
