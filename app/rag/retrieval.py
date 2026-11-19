"""Высокоуровневый поиск по фрагментам: режим, фильтры, порог.

Собирает репозиторий и эмбеддинги в одну операцию: кодирует запрос
(с префиксом query — это делает embedder), выбирает режим, прокидывает
фильтры по метаданным и порог релевантности.

CLI — для ручной проверки поиска без API:
    python -m app.rag.retrieval "мотор 800 Вт" --mode hybrid
"""
import argparse
import json
import sys

from app.core.config import settings
from app.core.db import connection
from app.rag import embedder
from app.repositories import chunks as chunks_repo

MODES = ("vector", "fulltext", "hybrid")


def search(
    query: str,
    *,
    mode: str = "hybrid",
    limit: int | None = None,
    min_score: float | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Выполнить поиск query в режиме mode, вернуть фрагменты с оценками.

    Каждый результат — dict: идентификаторы, текст, оценка (косинусная
    близость, ts_rank или вклад RRF — по режиму), метаданные. Порог
    min_score по умолчанию — из настроек: он калибруется на своих
    запросах, а не берётся с потолка.
    """
    if mode not in MODES:
        raise ValueError(f"неизвестный режим поиска: {mode}")
    if limit is None:
        limit = settings.retrieval_top_k
    if min_score is None:
        min_score = settings.retrieval_score_threshold
    with connection() as conn:
        if mode == "fulltext":
            return chunks_repo.search_fulltext(
                conn, query, limit=limit, min_score=min_score, filters=filters
            )
        query_vector = embedder.embed_query(query)
        if mode == "vector":
            return chunks_repo.search_vector(
                conn, query_vector, limit=limit, min_score=min_score, filters=filters
            )
        return chunks_repo.search_hybrid(
            conn, query_vector, query,
            limit=limit, min_score=min_score, filters=filters,
        )


def main() -> None:
    """Точка входа CLI: напечатать результаты поиска JSON-списком."""
    sys.stdout.reconfigure(encoding="utf-8")  # кириллица в консоли Windows

    parser = argparse.ArgumentParser(description="Поиск по фрагментам документов")
    parser.add_argument("query", help="поисковый запрос")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="режим поиска")
    parser.add_argument("--limit", type=int, default=None, help="сколько фрагментов вернуть")
    parser.add_argument("--min-score", type=float, default=None, help="порог релевантности")
    parser.add_argument("--document-id", default=None, help="фильтр по документу")
    parser.add_argument("--article", default=None, help="фильтр по артикулу")
    parser.add_argument("--brand", default=None, help="фильтр по бренду")
    args = parser.parse_args()

    filters = {
        key: value
        for key, value in {
            "document_id": args.document_id,
            "article": args.article,
            "brand": args.brand,
        }.items()
        if value is not None
    }
    results = search(
        args.query,
        mode=args.mode,
        limit=args.limit,
        min_score=args.min_score,
        filters=filters or None,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
