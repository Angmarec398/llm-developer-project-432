"""Проверка ссылок карточки — второй рубеж против выдумывания (шаг 7).

Первый рубеж — сам поиск: модель видит только то, что нашлось.
Второй — эта проверка: модель обязана вернуть источники, но может
приврать в ссылке, поэтому каждая проверяется по двум условиям:
фрагмент существует в базе и он действительно был в поданном
контексте. Второе условие важнее первого: сослаться на
существующий фрагмент, которого модель не видела, — выдумывание
с правдоподобным видом.

Нарушение отправляет карточку на переделку с текстом ошибки, при
повторе — к человеку. Решение об эскалации принимает workflow
(app/temporal/workflows.py); здесь только детерминированный факт.
"""
import psycopg

from app.repositories import chunks as chunks_repo
from app.schemas.cards import ProductCard


def verify_citations(
    conn: psycopg.Connection,
    card: ProductCard,
    context_chunk_ids: list[int],
) -> list[str]:
    """Проверить ссылки карточки, вернуть список нарушений.

    Пустой список — ссылки в порядке. Каждое нарушение — человеко-
    читаемая строка: она уходит генератору в feedback при переделке
    и в задачу при эскалации. Дубли ссылок на один фрагмент
    проверяются один раз: повтор не делает нарушение хуже.
    """
    known_ids = set(context_chunk_ids)
    violations: list[str] = []
    checked: set[int] = set()
    for source in card.sources:
        chunk_id = source.chunk_id
        if chunk_id in checked:
            continue
        checked.add(chunk_id)
        if chunks_repo.get_chunk(conn, chunk_id) is None:
            violations.append(f"фрагмент {chunk_id} не существует в базе")
        elif chunk_id not in known_ids:
            violations.append(
                f"фрагмент {chunk_id} не входил в поданный контекст"
            )
    return violations
