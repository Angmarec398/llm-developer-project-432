"""Парсер «грязного» JSON из ответов LLM.

Локальные модели любят оборачивать JSON в markdown-заборы и добавлять
болтовню до/после. Здесь одна точка, где это чистится.
"""
import json
from typing import Any


class LLMJsonError(ValueError):
    """Ответ не содержит распознаваемый JSON."""


def parse_llm_json(text: Any) -> dict:
    """Достать JSON-объект из ответа модели.

    Понимает: чистый JSON, JSON в ```-заборах, JSON с текстом вокруг,
    а также python-repr (title='...' вместо "title": ...) — локальные
    reasoning-модели иногда выдают именно его.
    Бросает LLMJsonError, если объекта не нашлось.
    """
    cleaned = str(text).strip()
    if not cleaned:
        raise LLMJsonError("пустой ответ модели")
    # Снятие markdown-заборов
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # JSON с болтовнёй вокруг: ищем внешний объект по скобкам
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # Python-repr: {"k": 'v'} / dict с одинарными кавычками
    if "='" in cleaned or "'=" in cleaned or "={'" in cleaned or "= '" in cleaned:
        import ast

        if start != -1 and end > start:
            try:
                parsed = ast.literal_eval(cleaned[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        # repr без скобок объекта целиком: title=... description=...
        try:
            parsed = ast.literal_eval("{" + cleaned + "}")
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    raise LLMJsonError(f"JSON не найден в ответе: {cleaned[:200]!r}")
