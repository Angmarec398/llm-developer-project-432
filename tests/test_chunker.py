"""Тесты резки на фрагменты (шаг 5): перекрытие, таблицы, метаданные."""
import itertools

from app.parsers.blocks import Block
from app.parsers.chunker import chunk_blocks

HEADER = "Артикул; мощность; объём"


def test_text_chunked_with_overlap():
    """Длинный текст режется окнами с перекрытием: граница не теряется."""
    text = " ".join(f"слово{number}" for number in range(200))
    chunks = chunk_blocks([Block(text=text, page=1)], size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    # перекрытие: соседние фрагменты делят общий кусок текста
    # (окна посимвольные, поэтому общая часть может резать слово)
    for left, right in itertools.pairwise(chunks):
        assert any(
            left.text[-length:] == right.text[:length] for length in range(5, 25)
        )


def test_short_text_single_chunk():
    """Короткий текст уходит одним фрагментом с метаданными блока."""
    block = Block(text="короткий текст", page=3, section="Обзор")
    chunks = chunk_blocks([block], size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "короткий текст"
    assert chunks[0].page == 3
    assert chunks[0].section == "Обзор"


def test_table_rows_never_split():
    """Строка таблицы не разрывается между фрагментами."""
    rows = [
        Block(
            text=f"ART-{number}: мощность {number}00 Вт; объём 1.5 л",
            kind="table_row",
            article=f"ART-{number}",
            table_header=HEADER,
        )
        for number in range(1, 11)
    ]
    chunks = chunk_blocks(rows, size=150, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        lines = chunk.text.split("\n")
        assert lines[0] == HEADER  # каждая часть несёт заголовки
        for line in lines[1:]:
            assert line.startswith("ART-") and line.endswith("л")


def test_table_fits_in_one_chunk():
    """Таблица, помещающаяся целиком, — один фрагмент с заголовком."""
    rows = [
        Block(
            text="BLD-800: мощность 800 Вт",
            kind="table_row",
            article="BLD-800",
            table_header=HEADER,
        )
    ]
    chunks = chunk_blocks(rows, size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == f"{HEADER}\nBLD-800: мощность 800 Вт"
    assert chunks[0].article == "BLD-800"


def test_order_and_mixed_blocks():
    """Текст и таблица идут в исходном порядке, номер страницы наследуется."""
    blocks = [
        Block(text="введение", page=1),
        Block(
            text="BLD-800: мощность 800 Вт",
            kind="table_row",
            article="BLD-800",
            table_header=HEADER,
        ),
        Block(text="заключение", page=2),
    ]
    chunks = chunk_blocks(blocks, size=800, overlap=150)
    assert [chunk.text.split("\n")[0] for chunk in chunks] == [
        "введение", HEADER, "заключение",
    ]
    assert chunks[2].page == 2
