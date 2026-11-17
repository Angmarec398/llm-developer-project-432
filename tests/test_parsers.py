"""Тесты разбора выданных файлов поставщиков (шаг 5).

Файлы в data/ сверяются автопроверкой по содержимому, поэтому тесты
ходят по реальным документам, а не по синтетическим фикстурам.
"""
from pathlib import Path

import pytest

from app.parsers.docx import parse_docx
from app.parsers.pdf import NoTextLayerError, parse_pdf
from app.parsers.xlsx import parse_xlsx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_pdf_pages_have_text_and_numbers():
    """Из pdf достаётся текст постранично, у каждого блока есть номер страницы."""
    blocks = parse_pdf(str(DATA_DIR / "blender_passport.pdf"))
    assert len(blocks) >= 2
    assert all(block.text.strip() for block in blocks)
    assert [block.page for block in blocks] == list(range(1, len(blocks) + 1))


def test_pdf_expected_content():
    """Разбор паспорта достаёт ожидаемые строки (бренд/мощность из документа)."""
    text = "\n".join(
        block.text for block in parse_pdf(str(DATA_DIR / "blender_passport.pdf"))
    )
    assert "BLD-800" in text
    assert "800" in text


def test_docx_sections_follow_headings():
    """В docx заголовок открывает секцию: блоки после него знают свою секцию."""
    blocks = parse_docx(str(DATA_DIR / "blender_kp.docx"))
    assert blocks
    assert all(block.page is None for block in blocks)
    assert any(block.section for block in blocks)
    # секция не может появиться раньше первого заголовка
    first_section_at = next(
        index for index, block in enumerate(blocks) if block.section
    )
    assert all(block.section is None for block in blocks[:first_section_at])


def test_xlsx_rows_become_article_lines():
    """Каждая строка спецификации читается как «артикул: параметр значение»."""
    blocks = parse_xlsx(str(DATA_DIR / "kettle_spec.xlsx"))
    assert blocks
    assert all(block.kind == "table_row" for block in blocks)
    for block in blocks:
        assert block.article
        assert block.text.startswith(f"{block.article}:")
        assert ";" in block.text  # параметров больше одного
        assert block.table_header  # заголовки колонок сохранены


def test_scanned_pdf_rejected_with_reason():
    """У отсканированного pdf нет текстового слоя — отказ, а не пустой результат."""
    with pytest.raises(NoTextLayerError):
        parse_pdf(str(DATA_DIR / "boiler_scan.pdf"))
