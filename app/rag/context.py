"""Сборка промпт-контекста из чанков retrieval.

Каждый чанк оформляется источником: [chunk_id | файл | стр. N | секция] —
фрагмент опознаваем по id, и модель обязана ссылаться на эти id.
Контекст сопровождается жёсткой инструкцией «отвечай только по контексту».
"""
import re

MAX_CONTEXT_CHARS = 6000  # ~1500 токенов — потолок контекста генерации

CONTEXT_INSTRUCTION = """Ты получаешь КОНТЕКСТ — фрагменты документов поставщика.
Правила:
1. Отвечай ТОЛЬКО на основе контекста. Нет данных — не выдумывай, добавь поле в missing_fields.
2. Каждое утверждение карточки подтверждай источником: chunk_id из скобки [chunk_id | ...].
3. В sources перечисли chunk_id, на которые опирался.
4. Уверенность (confidence) оценивай честно: мало данных — ниже.
"""


def build_context(chunks: list[dict], *, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Чанки → текст контекста с шапками-источниками, лимит по символам.

    Дедуп по chunk_id: один фрагмент в контексте один раз, даже если
    его нашли и векторный, и полнотекстовый поиск.
    """
    parts: list[str] = []
    total = 0
    seen: set[str] = set()
    for chunk in chunks:
        if chunk["chunk_id"] in seen:
            continue
        seen.add(chunk["chunk_id"])
        meta = chunk["metadata"]
        header = f"[{chunk['chunk_id']} | {chunk.get('filename', meta.get('doc_id', ''))}"
        if meta.get("page"):
            header += f" | стр. {meta['page']}"
        if meta.get("section"):
            header += f" | {meta['section']}"
        header += "]"
        body = header + "\n" + chunk["text"]
        if total + len(body) > max_chars and parts:
            break
        parts.append(body)
        total += len(body)
    return "\n\n".join(parts)


def context_chunk_ids(context: str) -> set[str]:
    """chunk_id, реально вошедшие в контекст (для валидации источников)."""
    return set(re.findall(r"^\[([a-z0-9]+) \|", context, flags=re.MULTILINE))
