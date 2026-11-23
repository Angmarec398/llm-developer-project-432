"""Тесты парсеров на файлах датасета: блоки, таблицы, отказ скана."""
from pathlib import Path

import pytest

from app.parsers.docx import parse_docx
from app.parsers.pdf import PdfNoTextLayer, parse_pdf
from app.parsers.xlsx import parse_xlsx
from app.parsers.normalizer import normalize

DATA = Path(__file__).resolve().parent.parent / "data"


def _read(name: str) -> bytes:
    """Прочитать файл датасета байтами."""
    return (DATA / name).read_bytes()


def test_pdf_passport_text_and_tables():
    """Паспорт блендера: текстовые блоки и две таблицы с строками."""
    blocks = parse_pdf(_read("blender_passport.pdf"))
    full = "\n".join(block.content for block in blocks)
    for probe in ("МиксерПро 800", "BLD-800", "800 Вт", "24 месяца"):
        assert probe in full, probe
    tables = [block for block in blocks if block.kind == "table"]
    assert len(tables) == 2
    assert len(tables[0].rows) == 13
    assert all(block.page >= 1 for block in blocks)


def test_scan_pdf_raises_no_text_layer():
    """Скан без текстового слоя — честный отказ, а не пустой список."""
    with pytest.raises(PdfNoTextLayer):
        parse_pdf(_read("boiler_scan.pdf"))


def test_docx_kp_blocks():
    """КП блендера: текст и одна таблица, контакты и цены на месте."""
    blocks = parse_docx(_read("blender_kp.docx"))
    full = "\n".join(block.content for block in blocks)
    for probe in ("BLD-800", "2 190", "+7 926 555-14-08"):
        assert probe in full, probe
    tables = [block for block in blocks if block.kind == "table"]
    assert len(tables) == 1


def test_xlsx_spec_rows_as_table_blocks():
    """Каждая строка спецификации — блок-таблица с артикулом в начале."""
    blocks = parse_xlsx(_read("kettle_spec.xlsx"))
    assert blocks
    assert all(block.kind == "table" for block in blocks)
    full = "\n".join(block.content for block in blocks)
    for articul in ("KTL-1700", "KTL-1000", "KTL-THRM"):
        assert articul in full, articul


def test_normalize_cleans_blocks():
    """normalize убирает колонтитулы и дефисы переносов из блоков датасета."""
    blocks = normalize(parse_pdf(_read("blender_passport.pdf")))
    full = "\n".join(block.content for block in blocks)
    assert "— стр." not in full
    assert "МиксерПро 800" in full
