"""Чанкер: fixed-size + overlap для текста, layout-aware для таблиц.

Metadata каждого чанка: {doc_id, page, section, articul, brand}.
Артикул/бренд вытаскиваются regex'ом из текста чанка.
"""
import re
import uuid

from app.parsers.pdf import Block

CHUNK_SIZE = 700      # символов
CHUNK_OVERLAP = 120

_ARTICUL_RE = re.compile(r"\b[A-Z]{2,5}-[A-Z0-9]{2,6}\b")
_BRANDS = ("МиксерПро", "КеттлПро", "АкваТерм", "ТехноДом")


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Блоки → чанки-словари с метаданными.

    Таблица — layout-aware: не рвём строку; большая таблица делится
    по строкам, каждая часть несёт контекст (шапку таблицы).
    """
    chunks: list[dict] = []
    for block in blocks:
        if block.kind == "table":
            pieces = _split_table(block, size)
            for piece in pieces:
                chunks.append(_make_chunk(piece, block, doc_id))
        else:
            for piece in _split_text(block.content, size, overlap):
                chunks.append(_make_chunk(piece, block, doc_id))
    for index, chunk in enumerate(chunks):
        chunk["ordinal"] = index
    return chunks


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Резка текста с перекрытием; слово посередине не рвём."""
    if len(text) <= size:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Параграф не рвём посередине слова — откатываемся до пробела
            cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [piece for piece in pieces if piece]


def _split_table(block: Block, size: int) -> list[str]:
    """Строку таблицы не рвём; режем между строками, каждая часть с заголовком."""
    lines = block.content.split("\n")
    header_line = lines[0] if lines else ""
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > size and current:
            pieces.append("\n".join(current))
            # Шапка таблицы дублируется в каждую часть — контекст колонок
            if header_line and not current[0].startswith(header_line.split(":")[0]):
                current = [f"[шапка] {header_line}", line]
                current_len = len(header_line) + len(line)
                continue
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


def _make_chunk(text: str, block: Block, doc_id: str) -> dict:
    """Собрать чанк-словарь: id, текст, метаданные (doc_id/page/section/артикул/бренд)."""
    meta = {
        "doc_id": doc_id,
        "page": block.page,
        "section": block.section or "",
    }
    articul = _ARTICUL_RE.search(text)
    if articul:
        meta["articul"] = articul.group(0)
    for brand in _BRANDS:
        if brand in text:
            meta["brand"] = brand
            break
    return {
        "id": uuid.uuid4().hex[:12],
        "text": text,
        "metadata": meta,
    }
