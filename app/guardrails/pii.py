"""PII-анонимизатор: телефоны, email, ИНН, карты — ДО подачи в LLM и в логи.

ИНН проверяется контрольной суммой, чтобы не маскировать случайные числа.
Сырые значения из модуля не возвращаются — только категория и плейсхолдер.
"""
import re
from dataclasses import dataclass

_PHONE_RE = re.compile(r"(?:\+7|8)[\s(-]?\d{3}[\s)-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_INN_RE = re.compile(r"\b\d{10}\b|\b\d{12}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass
class MaskedEntity:
    """Замаскированная сущность: kind (phone|email|inn|card) и плейсхолдер.

    original всегда пуст: сырые значения не покидают место маскирования.
    """
    kind: str
    placeholder: str
    original: str = ""


def _inn_checksum(digits: str) -> bool:
    """Контрольная сумма ИНН (10- и 12-значные)."""
    if len(digits) == 10:
        weights = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        return int(digits[-1]) == (sum(w * int(d) for w, d in zip(weights, digits)) % 11) % 10
    if len(digits) == 12:
        weights_first = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        weights_second = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        first = (sum(w * int(d) for w, d in zip(weights_first, digits)) % 11) % 10
        second = (sum(w * int(d) for w, d in zip(weights_second, digits)) % 11) % 10
        return int(digits[10]) == first and int(digits[11]) == second
    return False


def mask_pii(text: str) -> tuple[str, list[MaskedEntity]]:
    """Заменить PII на плейсхолдеры [PHONE_1], [EMAIL_1], [INN_1], [CARD_1].

    Возвращает (очищенный текст, список сущностей без исходников).
    """
    entities: list[MaskedEntity] = []
    counters = {"phone": 0, "email": 0, "inn": 0, "card": 0}
    result = text

    def _sub(pattern: re.Pattern, kind: str, check=None) -> None:
        nonlocal result

        def repl(match):
            if check and not check(match.group(0)):
                return match.group(0)
            counters[kind] += 1
            placeholder = f"[{kind.upper()}_{counters[kind]}]"
            entities.append(MaskedEntity(kind=kind, placeholder=placeholder))
            return placeholder

        result = pattern.sub(repl, result)

    _sub(_EMAIL_RE, "email")
    _sub(_PHONE_RE, "phone")
    _sub(
        _INN_RE,
        "inn",
        check=lambda value: _inn_checksum(re.sub(r"\D", "", value)),
    )
    _sub(_CARD_RE, "card", check=lambda value: len(re.sub(r"\D", "", value)) in (13, 16, 19))
    return result, entities


def contains_pii(text: str) -> bool:
    """Быстрая проверка для фильтра выхода: есть ли PII в тексте."""
    cleaned, found = mask_pii(text)
    return bool(found) and cleaned != text
