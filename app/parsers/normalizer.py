"""Нормализация текста: колонтитулы, дефисы на переносах, лигатуры, пробелы."""
import re
from copy import copy

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "­": "",  # soft hyphen
    " ": " ",  # nbsp
}


def clean_text(text: str) -> str:
    """Очистка одного блока: лигатуры, дефис-перенос, пробелы."""
    for char, replacement in _LIGATURES.items():
        text = text.replace(char, replacement)
    # Дефис на переносе строки: «мощ-\nность» → «мощность»
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Схлопывание пробелов
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_footers(blocks: list) -> list:
    """Удаление повторяющихся колонтитулов.

    Колонтитул — строка, встречающаяся на каждой странице документа
    (номер страницы в ней может отличаться, сравниваем по шаблону без цифр).
    Порог «каждая страница», а не «половина»: на коротком документе половина
    зачислит в колонтитулы любой обычный текст.
    """
    pages = {block.page for block in blocks}
    if len(pages) < 2:
        return blocks

    template_pages: dict[str, set[int]] = {}
    for block in blocks:
        if block.kind != "text":
            continue
        for line in block.content.split("\n"):
            if not line.strip():
                continue
            template = re.sub(r"\d+", "#", line.strip())
            template_pages.setdefault(template, set()).add(block.page)

    footers = {tpl for tpl, pgs in template_pages.items() if len(pgs) == len(pages)}

    result = []
    for block in blocks:
        if block.kind != "text":
            result.append(block)
            continue
        kept = [
            line for line in block.content.split("\n")
            if re.sub(r"\d+", "#", line.strip()) not in footers
        ]
        content = "\n".join(kept).strip()
        if content:
            stripped_block = copy(block)
            stripped_block.content = content
            result.append(stripped_block)
    return result


def normalize(blocks: list) -> list:
    """Полная нормализация: убрать колонтитулы, почистить текст, выкинуть пустое."""
    blocks = strip_footers(blocks)
    for block in blocks:
        block.content = clean_text(block.content)
    return [block for block in blocks if block.content]
