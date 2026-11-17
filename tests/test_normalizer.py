"""Тесты нормализатора (шаг 5): колонтитулы, переносы, лигатуры, пробелы."""
from app.parsers.blocks import Block
from app.parsers.normalizer import normalize_blocks


def test_running_header_removed():
    """Строка на каждой странице — колонтитул, она вычищается."""
    blocks = [
        Block(text=f"ACME Appliances\nТекст страницы {page}", page=page)
        for page in range(1, 5)
    ]
    normalized = normalize_blocks(blocks)
    assert len(normalized) == 4
    assert all("ACME Appliances" not in block.text for block in normalized)
    assert all("Текст страницы" in block.text for block in normalized)


def test_two_page_document_keeps_its_text():
    """На двухстраничном документе колонтитулы не вычищаются вовсе."""
    blocks = [
        Block(text="Общая строка\nТекст первой страницы", page=1),
        Block(text="Общая строка\nТекст второй страницы", page=2),
    ]
    normalized = normalize_blocks(blocks)
    assert [block.text for block in normalized] == [
        "Общая строка\nТекст первой страницы",
        "Общая строка\nТекст второй страницы",
    ]


def test_hyphenation_joined():
    """Перенос через дефис склеивается: «мощ-\\nность» → «мощность»."""
    blocks = [Block(text="мощ-\nность 800 Вт", page=1)]
    assert normalize_blocks(blocks)[0].text == "мощность 800 Вт"


def test_ligatures_and_spaces_collapsed():
    """Лигатуры раскладываются, множественные пробелы схлопываются."""
    blocks = [Block(text="\ufb01nal  \ufb02ow   chart", page=1)]
    assert normalize_blocks(blocks)[0].text == "final flow chart"


def test_empty_blocks_dropped():
    """Страница, состоявшая только из колонтитула, выпадает из результата."""
    blocks = [
        Block(text="ACME Appliances", page=page) for page in range(1, 4)
    ] + [Block(text="ACME Appliances\nСодержимое", page=4)]
    normalized = normalize_blocks(blocks)
    assert len(normalized) == 1
    assert normalized[0].text == "Содержимое"


def test_table_rows_untouched_by_running_lines():
    """Строки таблицы не проверяются на колонтитулы и не теряют артикул."""
    row = Block(
        text="BLD-800: мощность 800 Вт",
        kind="table_row",
        article="BLD-800",
        table_header="Артикул; мощность",
    )
    normalized = normalize_blocks([row])
    assert normalized[0].text == "BLD-800: мощность 800 Вт"
    assert normalized[0].article == "BLD-800"
