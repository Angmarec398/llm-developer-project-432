"""Тесты резки на фрагменты: перекрытие, таблицы не рвутся, метаданные."""
import itertools

from app.parsers.chunker import chunk_blocks
from app.parsers.pdf import Block


def test_text_chunked_with_overlap():
    """Длинный текст режется окнами с перекрытием: граница не теряется."""
    text = " ".join(f"слово{number}" for number in range(200))
    chunks = chunk_blocks(
        [Block("text", text, 1)], "doc1", size=100, overlap=20
    )
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 100 for chunk in chunks)
    # перекрытие: соседние фрагменты делят общий кусок текста
    for left, right in itertools.pairwise(chunks):
        assert any(
            left["text"][-length:] == right["text"][:length]
            for length in range(5, 25)
        )


def test_short_text_single_chunk():
    """Короткий текст уходит одним фрагментом с метаданными блока."""
    chunks = chunk_blocks(
        [Block("text", "короткий текст", 3, section="Обзор")], "doc1"
    )
    assert len(chunks) == 1
    assert chunks[0]["text"] == "короткий текст"
    assert chunks[0]["metadata"]["page"] == 3
    assert chunks[0]["metadata"]["section"] == "Обзор"
    assert chunks[0]["metadata"]["doc_id"] == "doc1"


def test_ordinals_assigned_in_order():
    """Порядковые номера фрагментов идут подряд с нуля."""
    text = " ".join(f"слово{number}" for number in range(200))
    chunks = chunk_blocks([Block("text", text, 1)], "doc1", size=100, overlap=20)
    assert [chunk["ordinal"] for chunk in chunks] == list(range(len(chunks)))


def test_chunk_ids_unique():
    """Каждый фрагмент получает свой идентификатор."""
    text = " ".join(f"слово{number}" for number in range(200))
    chunks = chunk_blocks([Block("text", text, 1)], "doc1", size=100, overlap=20)
    ids = [chunk["id"] for chunk in chunks]
    assert len(set(ids)) == len(ids)


def test_articul_extracted_to_metadata():
    """Артикул вида KTL-1700 поднимается в метаданные фрагмента."""
    chunks = chunk_blocks(
        [Block("text", "Чайник KTL-1700, мощность 2200 Вт", 1)], "doc1"
    )
    assert chunks[0]["metadata"].get("articul") == "KTL-1700"


def test_brand_extracted_to_metadata():
    """Известный бренд поднимается в метаданные фрагмента."""
    chunks = chunk_blocks(
        [Block("text", "Блендер МиксерПро 800, мощность 800 Вт", 1)], "doc1"
    )
    assert chunks[0]["metadata"].get("brand") == "МиксерПро"


def test_table_row_not_torn():
    """Строка таблицы не рвётся посередине: деление только между строками."""
    rows = ["Параметр: значение"] + [
        f"параметр{number}: {'x' * 60}" for number in range(20)
    ]
    block = Block("table", "\n".join(rows), 1, rows=rows)
    chunks = chunk_blocks([block], "doc1", size=200, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk["text"].split("\n"):
            if line.startswith("параметр"):
                assert line.endswith("x" * 60)  # строка целиком
