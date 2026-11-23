"""Парсер XLSX: openpyxl, строки спецификации → блоки «артикул: ключ значение»."""
import io

from openpyxl import load_workbook

from app.parsers.pdf import Block


def parse_xlsx(data: bytes, sheet: str | None = None) -> list[Block]:
    """Разобрать XLSX из байтов: каждая строка спецификации — один блок-таблица."""
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    rows = [
        [("" if cell is None else str(cell)).strip() for cell in row]
        for row in ws.iter_rows(values_only=True)
    ]
    wb.close()

    # Заголовок — первая непустая строка
    header_idx = next((i for i, row in enumerate(rows) if any(row)), None)
    if header_idx is None:
        return []
    header = rows[header_idx]

    blocks: list[Block] = []
    for row in rows[header_idx + 1:]:
        if not any(row):
            continue
        pairs = {h: v for h, v in zip(header, row) if h and v}
        articul = pairs.get("Артикул", "")
        # Строка спецификации → линейный текст «артикул: ключ: значение; …»
        body = "; ".join(f"{key}: {value}" for key, value in pairs.items())
        blocks.append(
            Block(
                kind="table",
                content=f"{articul}: {body}" if articul else body,
                page=1,
                section="Спецификация",
                rows=[list(pairs.values())],
            )
        )
    return blocks
