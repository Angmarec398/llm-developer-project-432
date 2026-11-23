"""Тесты нормализации: дефисы переносов, колонтитулы, лигатуры, пробелы."""
from app.parsers.normalizer import clean_text, normalize, strip_footers
from app.parsers.pdf import Block


def test_hyphenation_joined():
    """Дефис на переносе строки склеивается: «мощ-\\nность» → «мощность»."""
    assert clean_text("мощ-\nность") == "мощность"


def test_spaces_collapsed():
    """Повторные пробелы и табы схлопываются в один пробел."""
    assert clean_text("слева  справа") == "слева справа"
    assert clean_text("a\t\tb") == "a b"


def test_blank_runs_collapsed():
    """Три и больше перевода строки схлопываются до пустой строки."""
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_soft_hyphen_removed():
    """Мягкий перенос удаляется без следа."""
    assert clean_text("мощ­ность") == "мощность"


def test_footer_on_every_page_removed():
    """Строка-колонтитул на каждой странице удаляется, уникальный текст остаётся."""
    pages = ("комплектация и ножки", "мощность и скорости", "гарантия и сервис")
    blocks = [
        Block("text", f"Глава {page}: {body}\nООО «ТехноДом» — стр. {page}", page)
        for page, body in zip((1, 2, 3), pages)
    ]
    result = strip_footers(blocks)
    full = "\n".join(block.content for block in result)
    assert "ТехноДом" not in full
    assert "мощность и скорости" in full


def test_unique_lines_are_not_footers():
    """Строка на одной странице — не колонтитул, удалять нельзя."""
    blocks = [
        Block("text", "Гарантия 24 месяца", 1),
        Block("text", "Мощность 800 Вт", 2),
    ]
    result = strip_footers(blocks)
    assert [block.content for block in result] == [
        "Гарантия 24 месяца",
        "Мощность 800 Вт",
    ]


def test_single_page_document_untouched():
    """Одностраничный документ колонтитулов не имеет по определению."""
    blocks = [Block("text", "ООО «ТехноДом» — стр. 1", 1)]
    assert strip_footers(blocks) == blocks


def test_normalize_drops_empty_blocks():
    """Блоки, опустевшие после очистки, выбрасываются."""
    blocks = [
        Block("text", "  ", 1),
        Block("text", "Мощность 800 Вт", 1),
    ]
    result = normalize(blocks)
    assert len(result) == 1
    assert result[0].content == "Мощность 800 Вт"


def test_tables_pass_through_footer_strip():
    """Таблицы не разбираются на строки при чистке колонтитулов."""
    table = Block("table", "мощность: 800 Вт", 1)
    blocks = [table, Block("text", "стр. 1\nтекст", 1), Block("text", "стр. 2\nещё", 2)]
    result = strip_footers(blocks)
    assert table in result
