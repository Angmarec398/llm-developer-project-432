"""Разбор XLSX-спецификации «артикул → параметры».

Первая строка листа — заголовки колонок (обязательная часть шага:
число «800» без заголовка «мощность, Вт» ничего не значит). Каждая
следующая непустая строка превращается в текст вида
«BLD-800: мощность 800 Вт; чаша 1.5 л» — иначе поиск по смыслу
её не найдёт. Если в спецификации есть колонка «Бренд», её значение
уходит в метаданные бренда.
"""
from openpyxl import load_workbook

from app.parsers.blocks import Block


def parse_xlsx(file_path: str) -> list[Block]:
    """Превратить каждую строку спецификации в блок-table_row.

    Артикул — значение первой колонки; он же уходит в метаданные.
    Строка заголовков сохраняется в table_header: чанкер повторит
    её в начале каждой части большой таблицы. Страниц у XLSX нет,
    section заполняется именем листа.
    """
    workbook = load_workbook(file_path, read_only=True, data_only=True)
    blocks: list[Block] = []
    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        header_row = next(rows, None)
        if not header_row:
            continue
        headers = [str(cell).strip() if cell is not None else "" for cell in header_row]
        table_header = "; ".join(header for header in headers if header)
        brand_index = next(
            (index for index, header in enumerate(headers)
             if header.lower() == "бренд"),
            None,
        )
        for row in row_iter_values(rows):
            article, *parameter_values = row
            if article is None or not str(article).strip():
                continue
            parameters = [
                f"{header} {value}"
                for header, value in zip(headers[1:], parameter_values)
                if header and value is not None and str(value).strip()
            ]
            text = f"{str(article).strip()}: " + "; ".join(parameters)
            blocks.append(Block(
                text=text,
                kind="table_row",
                section=sheet.title,
                article=str(article).strip(),
                brand=(row[brand_index] if brand_index is not None else None),
                table_header=table_header,
            ))
    workbook.close()
    return blocks


def row_iter_values(rows) -> list[list]:
    """Привести строки листа к списку списков значений (пропуская пустые)."""
    normalized = []
    for row in rows:
        values = [str(cell).strip() if cell is not None else None for cell in row]
        if any(value for value in values):
            normalized.append(values)
    return normalized
