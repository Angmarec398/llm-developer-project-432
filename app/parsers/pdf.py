"""Разбор PDF: текст извлекается постранично.

Отдельно обрабатывается отсканированный документ: текстового слоя
в нём нет, и молча вернуть пустой результат нельзя — пустая карточка
без объяснения выглядит как поломка сервиса. Поэтому парсер бросает
NoTextLayerError, а сервис ставит документу отказ с причиной.
"""
from pypdf import PdfReader

from app.parsers.blocks import Block


class NoTextLayerError(Exception):
    """В PDF нет текстового слоя (скан) — без OCR разбор невозможен."""


def parse_pdf(file_path: str) -> list[Block]:
    """Извлечь текст из PDF постранично.

    Каждая непустая страница становится блоком со своим номером
    (нумерация с 1 — для ссылки на источник). Если текста нет ни на
    одной странице, бросает NoTextLayerError.
    """
    reader = PdfReader(file_path)
    blocks: list[Block] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append(Block(text=text, page=page_number))
    if not blocks:
        raise NoTextLayerError(f"в файле нет текстового слоя: {file_path}")
    return blocks
