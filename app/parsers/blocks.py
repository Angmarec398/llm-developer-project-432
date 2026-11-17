"""Общие структуры данных для парсеров и чанкера.

Block — структурная единица документа после разбора: каждый парсер
отдаёт список блоков, чанкер режет их на фрагменты, наследуя
метаданные (страница, секция, артикул, бренд).
"""
from dataclasses import dataclass


@dataclass
class Block:
    """Структурная единица документа после разбора."""

    text: str
    kind: str = "text"  # "text" | "table_row"
    page: int | None = None
    section: str | None = None
    article: str | None = None
    brand: str | None = None
    table_header: str | None = None  # строка заголовков (для kind="table_row")
