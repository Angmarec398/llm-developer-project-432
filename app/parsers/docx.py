"""Разбор DOCX: параграфы, заголовок работает границей секции.

Страниц у DOCX нет (вёрстка определяется при открытии), поэтому
page всегда None; привязка к месту документа идёт через section.
"""
from docx import Document

from app.parsers.blocks import Block


def parse_docx(file_path: str) -> list[Block]:
    """Извлечь параграфы из DOCX с привязкой к секциям.

    Параграф со стилем Heading открывает новую секцию: его текст
    попадает в метаданные section всех последующих параграфов —
    по нему потом видно, из какой части документа взят фрагмент.
    Пустые параграфы пропускаются.
    """
    document = Document(file_path)
    blocks: list[Block] = []
    section: str | None = None
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.startswith("Heading"):
            section = text
        blocks.append(Block(text=text, section=section))
    return blocks
