"""Резка нормализованных блоков документа на фрагменты (чанки).

Два режима из PDF шага 5:
- текст режется фиксированным размером с перекрытием — утверждение,
  попавшее на границу фрагментов, не теряется;
- таблица режется по строкам: строку рвать нельзя, и каждая часть
  большой таблицы начинается со строки заголовков колонок — иначе
  «800» без «мощность, Вт» ничего не значит.

Метаданные (страница, секция, артикул, бренд) наследуются от блока;
по ним потом фильтруется поиск и строится ссылка на источник.
"""
from dataclasses import dataclass

from app.parsers.blocks import Block


@dataclass
class Chunk:
    """Фрагмент документа, готовый к сохранению в базу."""

    text: str
    page: int | None = None
    section: str | None = None
    article: str | None = None
    brand: str | None = None


def chunk_blocks(blocks: list[Block], size: int, overlap: int) -> list[Chunk]:
    """Разрезать последовательность блоков на фрагменты.

    Порядок блоков сохраняется: фрагменты выходят в том же порядке,
    в каком блоки шли в документе (это потом станет полем seq в БД).
    Перекрытие должно быть меньше размера фрагмента, иначе шаг
    не положительный и резка не сходится.
    """
    if overlap >= size:
        raise ValueError("перекрытие должно быть меньше размера фрагмента")
    chunks: list[Chunk] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block.kind == "table_row":
            table_rows, index = _take_table(blocks, index)
            chunks.extend(_chunk_table(table_rows, size))
        else:
            chunks.extend(_chunk_text(block, size, overlap))
            index += 1
    return chunks


def _take_table(blocks: list[Block], start: int) -> tuple[list[Block], int]:
    """Забрать подряд идущие строки одной таблицы, начиная с позиции start."""
    header = blocks[start].table_header
    table_rows: list[Block] = []
    index = start
    while index < len(blocks):
        block = blocks[index]
        if block.kind != "table_row" or block.table_header != header:
            break
        table_rows.append(block)
        index += 1
    return table_rows, index


def _chunk_text(block: Block, size: int, overlap: int) -> list[Chunk]:
    """Разрезать текстовый блок окнами длиной size с шагом size - overlap."""
    text = block.text
    if len(text) <= size:
        return [_inherit(block, text)]
    chunks: list[Chunk] = []
    step = size - overlap
    position = 0
    while position < len(text):
        piece = text[position : position + size].strip()
        if piece:
            chunks.append(_inherit(block, piece))
        position += step
    return chunks


def _chunk_table(table_rows: list[Block], size: int) -> list[Chunk]:
    """Разрезать таблицу, не разрывая строки и повторяя заголовки.

    Новая часть начинается, когда очередная строка не помещается
    в текущий фрагмент; каждая часть открывается строкой заголовков.
    Строка длиннее size уходит отдельным фрагментом — строку
    рвать нельзя даже ценой превышения размера.
    """
    header = table_rows[0].table_header or ""
    chunks: list[Chunk] = []
    current_lines = [header]
    current_first_row = table_rows[0]
    for row in table_rows:
        candidate = "\n".join([*current_lines, row.text])
        if len(candidate) > size and len(current_lines) > 1:
            chunks.append(_inherit(current_first_row, "\n".join(current_lines)))
            current_lines = [header, row.text]
            current_first_row = row
        else:
            current_lines.append(row.text)
    if len(current_lines) > 1:
        chunks.append(_inherit(current_first_row, "\n".join(current_lines)))
    return chunks



def _inherit(block: Block, text: str) -> Chunk:
    """Собрать фрагмент из текста, унаследовав метаданные блока."""
    return Chunk(
        text=text,
        page=block.page,
        section=block.section,
        article=block.article,
        brand=block.brand,
    )
