"""Сборка контекста из результатов поиска (шаг 7).

Контекст — единственное, что модель видит про товар: генерация
работает от найденных фрагментов, а не от целого текста. Поэтому
каждый фрагмент в блоке опознаваем — у него явный идентификатор,
на который карточка обязана сослаться, а приложение потом проверяет
ссылку (фрагмент есть в базе и был именно в этом контексте).

Два ограничения сборки:
- повторы: одинаковый текст из разных документов включается один
  раз (первое вхождение по рангу выдачи) — иначе бюджет контекста
  уходит на дубли;
- предел размера: блок не длиннее context_max_chars символов,
  хвост выдачи отсекается первым.
"""
from app.core.config import settings


def build_context(
    results: list[dict], max_chars: int | None = None
) -> tuple[str, list[int]]:
    """Собрать блок контекста и id фрагментов, реально вошедших в него.

    Фрагменты идут в порядке выдачи (лучший первым). Каждый получает
    ярлык с id и строку-источник — по id карточка ссылается на
    источник. Дубли по тексту пропускаются; если блок превышает
    max_chars (по умолчанию — из настроек), хвост выдачи отсекается.
    Второе возвращаемое значение — id вошедших фрагментов: на этом
    списке проверяется, что ссылка карточки была в поданном контексте
    (app/services/citations.py). Пустой поиск — честно пустая строка
    и пустой список: мусор из заглушек в промпт не попадает,
    вызывающий код решает, что делать без контекста.
    """
    if max_chars is None:
        max_chars = settings.context_max_chars
    blocks: list[str] = []
    included_ids: list[int] = []
    seen_texts: set[str] = set()
    used_chars = 0
    for chunk in results:
        text = chunk["text"]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        block = (
            f"[{_block_label(chunk, len(blocks) + 1)}] "
            f"{_source_line(chunk)}\n{text}"
        )
        separator_len = 2 if blocks else 0
        if blocks and used_chars + separator_len + len(block) > max_chars:
            break
        used_chars += separator_len + len(block)
        blocks.append(block)
        if chunk.get("id") is not None:
            included_ids.append(chunk["id"])
    return "\n\n".join(blocks), included_ids


def format_context(results: list[dict], max_chars: int | None = None) -> str:
    """Только текстовый блок контекста, без id (отладка, CLI-прогоны)."""
    context_text, _included_ids = build_context(results, max_chars)
    return context_text


def _block_label(chunk: dict, ordinal: int) -> str:
    """Ярлык блока, который модель копирует в chunk_id.

    Ярлык равен id фрагмента — тому, что карточка должна вернуть
    в sources: модель копирует его, а не выбирает число поменьше.
    Позиция в выдаче и id в базе — разные значения, смешение их
    даёт «правдоподобную» ссылку вне контекста. У фрагмента без id
    (отладочные вызовы) — номер позиции, он ни на что не ссылается.
    """
    if chunk.get("id") is not None:
        return f"id {chunk['id']}"
    return f"позиция {ordinal}"


def _source_line(chunk: dict) -> str:
    """Строка-источник из заполненных метаданных фрагмента."""
    parts = [
        f"документ {chunk['document_id']}",
        f"фрагмент {chunk['seq']}",
    ]
    if chunk.get("id") is not None:
        parts.append(f"id {chunk['id']}")
    if chunk.get("page") is not None:
        parts.append(f"стр. {chunk['page']}")
    if chunk.get("section"):
        parts.append(f"раздел «{chunk['section']}»")
    if chunk.get("article"):
        parts.append(f"артикул {chunk['article']}")
    if chunk.get("brand"):
        parts.append(f"бренд {chunk['brand']}")
    return ", ".join(parts)
