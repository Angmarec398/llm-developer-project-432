"""Тесты сборки контекста: шапки-источники, дедуп, лимит, извлечение id."""
from app.rag.context import build_context, context_chunk_ids


def _chunk(chunk_id: str, text: str, page: int = 1, section: str = "") -> dict:
    """Фрагмент выдачи поиска с метаданными."""
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc1",
        "text": text,
        "metadata": {"doc_id": "doc1", "page": page, "section": section},
        "filename": "spec.pdf",
    }


def test_headers_make_chunks_identifiable():
    """Каждый фрагмент опознаваем: id, файл, страница и секция в шапке."""
    context = build_context([
        _chunk("abc123def456", "Мощность 800 Вт", page=2, section="Характеристики")
    ])
    header = context.splitlines()[0]
    assert "abc123def456" in header
    assert "spec.pdf" in header
    assert "2" in header
    assert "Характеристики" in header
    assert "Мощность 800 Вт" in context


def test_duplicates_removed_by_chunk_id():
    """Один и тот же фрагмент не попадает в контекст дважды."""
    chunk = _chunk("abc123def456", "текст")
    context = build_context([chunk, dict(chunk)])
    assert context.count("abc123def456") == 1


def test_max_chars_limit():
    """Жёсткий потолок: фрагмент, не влезающий целиком, не берётся."""
    chunks = [_chunk(f"chunk{i:08d}", "x" * 4000) for i in range(5)]
    context = build_context(chunks, max_chars=6000)
    assert len(context) < 6000 + 100


def test_context_chunk_ids_extracted():
    """Из собранного контекста восстанавливаются id вошедших фрагментов."""
    chunks = [_chunk("aaa111bbb222", "первый"), _chunk("ccc333ddd444", "второй")]
    context = build_context(chunks)
    assert context_chunk_ids(context) == {"aaa111bbb222", "ccc333ddd444"}


def test_empty_chunks_empty_context():
    """Пустая выдача — пустой контекст без id."""
    assert build_context([]) == ""
    assert context_chunk_ids("") == set()
