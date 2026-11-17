"""Нормализация текста блоков после разбора.

Четыре правила из PDF шага 5:
- колонтитул — строка, повторяющаяся на страницах; ловушка: правило
  «повторяется на большинстве страниц» на двухстраничном документе
  выметает весь текст, потому что большинство от двух — это одна,
  поэтому колонтитулом считается строка, встречающаяся минимум
  на трёх страницах и более чем на половине страниц;
- перенос через дефис склеивается обратно, иначе «мощ-ность»
  не найдётся никаким поиском;
- лигатуры (ﬁ, ﬂ, ﬀ и т.п.) раскладываются в обычные буквы;
- лишние пробелы и пустые строки схлопываются.
"""
import re
from collections import Counter
from dataclasses import replace

from app.parsers.blocks import Block

MIN_RUNNING_LINE_PAGES = 3

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

_DEHYPHENATION = re.compile(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_blocks(blocks: list[Block]) -> list[Block]:
    """Нормализовать текст блоков, вернуть новый список.

    Порядок важен: сначала вычищаются колонтитулы (они целые строки),
    потом склейка переносов (она работает через границу строки),
    затем лигатуры и пробелы. Блоки, опустевшие после чистки,
    из результата выпадают. Исходные блоки не изменяются.
    """
    running_lines = _detect_running_lines(blocks)
    normalized: list[Block] = []
    for block in blocks:
        text = block.text
        if block.kind == "text" and running_lines:
            text = "\n".join(
                line
                for line in text.split("\n")
                if line.strip() not in running_lines
            )
        text = _normalize_text(text)
        if text:
            normalized.append(replace(block, text=text))
    return normalized


def _detect_running_lines(blocks: list[Block]) -> set[str]:
    """Найти строки-колонтитулы: повторяются на большинстве страниц.

    Учитываются только постраничные текстовые блоки (pdf). Порог —
    минимум MIN_RUNNING_LINE_PAGES страниц и строго больше половины
    страниц: защита от выметания всего текста на коротких документах.
    """
    paged_blocks = [
        block for block in blocks if block.kind == "text" and block.page is not None
    ]
    total_pages = len(paged_blocks)
    if total_pages < MIN_RUNNING_LINE_PAGES:
        return set()
    line_pages: Counter[str] = Counter()
    for block in paged_blocks:
        for line in {line.strip() for line in block.text.split("\n") if line.strip()}:
            line_pages[line] += 1
    return {
        line
        for line, pages_count in line_pages.items()
        if pages_count >= MIN_RUNNING_LINE_PAGES and pages_count > total_pages / 2
    }


def _normalize_text(text: str) -> str:
    """Склеить переносы, разложить лигатуры, схлопнуть пробелы."""
    text = _DEHYPHENATION.sub(r"\1\2", text)
    for ligature, expansion in LIGATURES.items():
        text = text.replace(ligature, expansion)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
