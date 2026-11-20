"""Тесты сборки контекста: опознаваемость, дедуп, предел размера."""
from app.rag.context import format_context


def _chunk(seq: int, text: str, **overrides) -> dict:
    """Фрагмент поиска с разумными умолчаниями и переопределениями."""
    chunk = {
        "id": seq,
        "document_id": f"doc-{seq}",
        "seq": seq,
        "text": text,
        "page": None,
        "section": None,
        "article": None,
        "brand": None,
        "score": 0.9,
    }
    chunk.update(overrides)
    return chunk


def test_every_chunk_identifiable():
    """У каждого фрагмента в контексте есть идентификатор для ссылки."""
    result = format_context([_chunk(1, "Мощность 2200 Вт")])
    assert "id 1" in result
    assert "документ doc-1" in result
    assert "Мощность 2200 Вт" in result


def test_duplicate_text_included_once():
    """Одинаковый текст из разных документов попадает в контекст один раз."""
    results = [
        _chunk(1, "Гарантия 2 года", document_id="doc-a"),
        _chunk(2, "Гарантия 2 года", document_id="doc-b"),
        _chunk(3, "Объём 1.7 л", document_id="doc-c"),
    ]
    result = format_context(results)
    assert result.count("Гарантия 2 года") == 1
    assert "Объём 1.7 л" in result


def test_context_size_limit():
    """Блок не длиннее лимита: хвост выдачи отсекается первым."""
    results = [
        _chunk(index, f"текст {index} " + "x" * 100) for index in range(1, 6)
    ]
    result = format_context(results, max_chars=250)
    assert len(result) <= 250
    assert "текст 1" in result
    assert "текст 5" not in result


def test_empty_search_gives_empty_context():
    """Пустой поиск — честно пустая строка, без заглушек."""
    assert format_context([]) == ""
