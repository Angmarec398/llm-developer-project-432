"""Детекторы персональных данных и маскирование (шаг 9).

Первый рубеж: ПДн из документов поставщиков не должны уйти ни в чужое
API, ни в свои логи. Детекторы типовых категорий — телефон, почта,
номер налогоплательщика, номер банковской карты — заменяют найденное
метками вида «телефон номер один». Работают на фрагменте при сборке
контекста: до любого обращения к модели и до записи контекста в лог
или историю процесса.

Два правила, которые нельзя пропустить:

- Номер налогоплательщика проверяется контрольной суммой. По одной
  длине детектор маскировал бы всё подряд — артикулы, суммы, даты.
  Защита, ломающая данные, — отказ такой же, как пропущенная атака,
  только заметить его труднее.
- Номер карты проверяется алгоритмом Луна по той же причине.

Сырые значения никуда не возвращаются: находка — категория и позиция,
метка строится при замене. Поэтому отчёт безопасно кладётся в логи,
базу и результат задачи.
"""
import re
from dataclasses import dataclass

# Категория -> слово для метки. Порядок кортежа — приоритет при
# пересечении кандидатов: телефон из 11 цифр не должен сожрать кусок
# 16-значного номера карты раньше проверки Луна.
_DETECTORS = (
    ("phone", "телефон"),
    ("email", "электронная почта"),
    ("taxpayer_number", "номер налогоплательщика"),
    ("card_number", "номер карты"),
)
_CATEGORY_LABEL = dict(_DETECTORS)

# +7/8, пробелы, скобки, дефисы; ровно 10 цифр после кода страны.
# Границы (?!\\d) не дают сожрать кусок длинной последовательности.
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"
)
_EMAIL_PATTERN = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])"
)
# 10 цифр (юрлицо) или 12 (физлицо/ИП) не в составе более длинной
# последовательности; годится только с валидной контрольной суммой.
_TAXPAYER_PATTERN = re.compile(r"(?<!\d)(?:\d{10}|\d{12})(?!\d)")
# 13–19 цифр с пробелами/дефисами между группами; годится только по Луна.
_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

_INN_WEIGHTS_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_WEIGHTS_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)

_NUMBER_WORDS = {
    1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
    6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
}


@dataclass
class PIIFinding:
    """Найденное значение ПДн: категория и позиция в исходном тексте.

    Само значение намеренно не хранится — единственная копия остаётся
    во входном тексте, который вызывающий код обязан не логировать.
    """

    category: str   # ключ из _DETECTORS: phone | email | ...
    start: int      # смещение начала вхождения в тексте
    end: int        # смещение конца вхождения (не включая)


@dataclass
class MaskingResult:
    """Результат маскирования: новый текст и список находок."""

    text: str
    findings: list[PIIFinding]


def detect_pii(text: str) -> list[PIIFinding]:
    """Найти ПДн в тексте, отсортировано по позиции, без пересечений.

    Кандидаты проходят проверку категории: контрольная сумма для номера
    налогоплательщика, Луна для номера карты. Невалидные отбрасываются
    — артикул из 10 цифр находкой не считается. При пересечении
    кандидатов выигрывает более приоритетная категория (порядок
    _DETECTORS): телефон целиком, а не его 11-значный хвост как «карта».
    """
    return _collect(text)


def mask_pii(text: str) -> MaskingResult:
    """Заменить найденное ПДн метками вида «телефон номер один».

    Метки нумеруются внутри каждой категории по порядку появления.
    Без находок возвращает исходный текст как есть — вызывающий код
    не обязан проверять findings, прежде чем использовать text.
    """
    findings = detect_pii(text)
    if not findings:
        return MaskingResult(text=text, findings=[])
    counters: dict[str, int] = {}
    parts: list[str] = []
    cursor = 0
    for finding in findings:
        parts.append(text[cursor:finding.start])
        counters[finding.category] = counters.get(finding.category, 0) + 1
        parts.append(_finding_label(finding.category, counters[finding.category]))
        cursor = finding.end
    parts.append(text[cursor:])
    return MaskingResult(text="".join(parts), findings=findings)


def masked_labels(findings: list[PIIFinding]) -> list[tuple[str, str]]:
    """Пары (категория, метка) для отчёта: что именно замаскировано.

    Метки восстанавливаются той же нумерацией, что и при замене, —
    отчёт описывает текст, который реально ушёл в промпт.
    """
    counters: dict[str, int] = {}
    labels: list[tuple[str, str]] = []
    for finding in findings:
        counters[finding.category] = counters.get(finding.category, 0) + 1
        labels.append(
            (finding.category, _finding_label(finding.category, counters[finding.category]))
        )
    return labels


def _pattern_for(category: str) -> re.Pattern:
    """Regex детектора по категории."""
    return {
        "phone": _PHONE_PATTERN,
        "email": _EMAIL_PATTERN,
        "taxpayer_number": _TAXPAYER_PATTERN,
        "card_number": _CARD_PATTERN,
    }[category]


def _value_confirmed(category: str, raw_value: str) -> bool:
    """Проверка кандидата правилами категории перед зачётом находки."""
    if category == "taxpayer_number":
        return _taxpayer_checksum_ok(raw_value)
    if category == "card_number":
        digits = re.sub(r"\D", "", raw_value)
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    return True


def _collect(text: str) -> list[PIIFinding]:
    """Собрать непересекающиеся находки всех детекторов по приоритету."""
    candidates: list[tuple[int, int, int, PIIFinding]] = []
    for priority, (category, _label) in enumerate(_DETECTORS):
        for match in _pattern_for(category).finditer(text):
            if not _value_confirmed(category, match.group()):
                continue
            candidates.append(
                (match.start(), priority, match.end(), PIIFinding(category, match.start(), match.end()))
            )
    candidates.sort(key=lambda item: (item[0], item[1]))
    accepted: list[PIIFinding] = []
    occupied_until = 0
    for start, _priority, end, finding in candidates:
        if start < occupied_until:
            continue
        accepted.append(finding)
        occupied_until = end
    return accepted


def _finding_label(category: str, ordinal: int) -> str:
    """Метка замены: «телефон номер один», дальше десяти — цифрой."""
    word = _NUMBER_WORDS.get(ordinal, str(ordinal))
    return f"{_CATEGORY_LABEL[category]} номер {word}"


def _taxpayer_checksum_ok(digits: str) -> bool:
    """Контрольная сумма ИНН: 10-значный — одна проверочная цифра,
    12-значный — две (11-я и 12-я). Веса — из спецификации ФНС."""
    if len(digits) == 10:
        expected = sum(
            int(digit) * weight for digit, weight in zip(digits[:9], _INN_WEIGHTS_10)
        ) % 11 % 10
        return expected == int(digits[9])
    if len(digits) == 12:
        first = sum(
            int(digit) * weight for digit, weight in zip(digits[:10], _INN_WEIGHTS_11)
        ) % 11 % 10
        second = sum(
            int(digit) * weight for digit, weight in zip(digits[:11], _INN_WEIGHTS_12)
        ) % 11 % 10
        return first == int(digits[10]) and second == int(digits[11])
    return False


def _luhn_ok(digits: str) -> bool:
    """Алгоритм Луна: контрольная сумма номера банковской карты."""
    total = 0
    for index, digit_char in enumerate(reversed(digits)):
        digit = int(digit_char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
