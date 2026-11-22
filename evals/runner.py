"""Раннер эталонных прогонов (шаг 8): генерация по golden_cards.json.

Для каждого документа эталона: документ находится (или создаётся) в
базе, доразбирается/доиндексируется, собирается контекст (запрос —
название продукта из эталона), модель генерирует одну карточку,
метрики сравнивают её с ожиданиями. Критик и цикл переделок сюда не
входят: метрики меряют генерацию, а не согласование. Истинность
оценивает модель-судья (служебная — политика двух моделей).

Прогон по умолчанию идёт по eval_defaults (быстрое подмножество),
флаг full — весь набор. Отчёт печатается и сохраняется файлом:
без файла сравнить «до» и «после» правки промпта не с чем. Вызовы
модели попадают в общий учёт llm_calls (job_id пуст — это не задача
пользователя).
"""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from app.core.db import connection
from app.core.logging import get_logger
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.services import cards as cards_service
from app.services import pipeline
from evals.metrics import (
    characteristics_score,
    citation_probes_score,
    faithfulness_score,
)

log = get_logger(__name__)

GOLDEN_PATH = Path("data/golden_cards.json")
REPORT_DIR = Path("evals")


async def run_eval(full: bool = False) -> Path:
    """Прогнать метрики по эталону, сохранить отчёт, вернуть путь к нему."""
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    filenames = list(golden["documents"]) if full else golden["eval_defaults"]
    rows = []
    for filename in filenames:
        log.info("eval_document_start", document=filename)
        rows.append(await evaluate_document(filename, golden["documents"][filename]))
    report_path = _save_report(rows, full)
    _print_report(rows, report_path)
    return report_path


async def evaluate_document(filename: str, entry: dict) -> dict:
    """Одна строка отчёта: метрики карточки документа против эталона."""
    if not entry.get("characteristics") and not entry.get("source_probes"):
        return {"document": filename, "skipped": entry.get("notes", "нет ожиданий")}
    document_id = _ensure_document(filename)
    cards_service.parse_pending_documents([document_id])
    cards_service.embed_pending_documents([document_id])
    results, _report = cards_service.search_chunks(
        [document_id], hint=entry.get("product")
    )
    context, _chunk_ids = cards_service.assemble_context(results)
    card = await pipeline.generate_card(context)
    verdict = await pipeline.judge_faithfulness(context, card)
    chunks_by_id = _card_chunks(card)
    return {
        "document": filename,
        "title": card.title,
        "confidence": card.confidence,
        "characteristics": characteristics_score(
            entry.get("characteristics", {}), card.characteristics
        ),
        "citation": citation_probes_score(
            entry.get("source_probes", {}),
            [source.model_dump() for source in card.sources],
            chunks_by_id,
        ),
        "faithfulness": faithfulness_score(verdict),
    }


def _ensure_document(filename: str) -> str:
    """Найти документ эталона в базе по хешу, при отсутствии — создать."""
    file_path = Path("data") / filename
    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    with connection() as conn:
        document_id, _created = documents_repo.create_document(
            conn, filename, str(file_path), content_hash
        )
    return document_id


def _card_chunks(card) -> dict[int, dict]:
    """Фрагменты-источники карточки из базы, по id — для проб цитирования."""
    chunks = {}
    with connection() as conn:
        for source in card.sources:
            chunk = chunks_repo.get_chunk(conn, source.chunk_id)
            if chunk is not None:
                chunks[chunk["id"]] = chunk
    return chunks


def _print_report(rows: list[dict], report_path: Path) -> None:
    """Таблица по документам и средние — на экран."""
    print()
    header = (
        f"{'документ':<24} {'характеристики':>16} "
        f"{'цитирование':>12} {'истинность':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if "skipped" in row:
            print(f"{row['document']:<24} пропущен: {row['skipped']}")
            continue
        print(
            f"{row['document']:<24} "
            f"{row['characteristics']:>16.2f} "
            f"{row['citation']:>12.2f} "
            f"{row['faithfulness']:>12.2f}"
        )
    scored = [row for row in rows if "characteristics" in row]
    if scored:
        print("-" * len(header))
        print(
            f"{'среднее':<24} "
            f"{_average(scored, 'characteristics'):>16.2f} "
            f"{_average(scored, 'citation'):>12.2f} "
            f"{_average(scored, 'faithfulness'):>12.2f}"
        )
    print(f"\nотчёт: {report_path}")


def _average(rows: list[dict], key: str) -> float:
    """Среднее по ключу для непустых строк отчёта."""
    return sum(row[key] for row in rows) / len(rows)


def _save_report(rows: list[dict], full: bool) -> Path:
    """Сохранить отчёт файлом — иначе «до/после» правки промпта не сравнить."""
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"report_{stamp}.md"
    scored = [row for row in rows if "characteristics" in row]
    lines = [
        "# Метрики качества генерации",
        "",
        f"- дата: {datetime.now(UTC).isoformat()}",
        f"- набор: {'полный' if full else 'eval_defaults'}",
        "",
        "| документ | характеристики | цитирование | истинность |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if "skipped" in row:
            lines.append(f"| {row['document']} | — (пропущен) | — | — |")
            continue
        lines.append(
            f"| {row['document']} | {row['characteristics']:.2f} "
            f"| {row['citation']:.2f} | {row['faithfulness']:.2f} |"
        )
    if scored:
        lines.append(
            f"| **среднее** | **{_average(scored, 'characteristics'):.2f}** "
            f"| **{_average(scored, 'citation'):.2f}** "
            f"| **{_average(scored, 'faithfulness'):.2f}** |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
