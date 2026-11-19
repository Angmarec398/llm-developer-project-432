"""Сборка контекста из результатов поиска.

Поиск возвращает фрагменты; здесь они превращаются в текстовый
блок для промпта (шаг 7 подключит его к генерации карточки).
Каждый фрагмент — с источником: документ, страница, раздел,
артикул, бренд. Модель должна видеть, откуда факт, чтобы
сгенерированная карточка могла сослаться на первоисточник.
"""
_CONTEXT_COUNTER_START = 1


def format_context(results: list[dict]) -> str:
    """Превратить список фрагментов поиска в текстовый блок контекста.

    Фрагменты идут в порядке выдачи поиска (лучший первым), каждый
    с номером и строкой-источником. Пустой поиск — честно пустая
    строка: вызывающий код решает, что делать без контекста, мусор
    из заглушек в промпт не попадает.
    """
    blocks = []
    for position, chunk in enumerate(results, start=_CONTEXT_COUNTER_START):
        blocks.append(f"[{position}] {_source_line(chunk)}\n{chunk['text']}")
    return "\n\n".join(blocks)


def _source_line(chunk: dict) -> str:
    """Строка-источник из заполненных метаданных фрагмента."""
    parts = [
        f"документ {chunk['document_id']}",
        f"фрагмент {chunk['seq']}",
    ]
    if chunk.get("page") is not None:
        parts.append(f"стр. {chunk['page']}")
    if chunk.get("section"):
        parts.append(f"раздел «{chunk['section']}»")
    if chunk.get("article"):
        parts.append(f"артикул {chunk['article']}")
    if chunk.get("brand"):
        parts.append(f"бренд {chunk['brand']}")
    return ", ".join(parts)
