"""Нормализация значений для сравнения с эталоном (шаг 8).

Часть метрики, а не деталь: «800 Вт», «800 вт» и «800 Вт.» — одно
и то же значение. Сравнение буквальным равенством занизило бы
characteristics_score и скрывало бы реальные регрессии.
"""
import re
import unicodedata

_DASHES = str.maketrans({
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # минус
})

_UNIT_SYNONYMS = {
    "ватт": "вт",
    "литр": "л",
    "литра": "л",
    "грамм": "г",
    "герц": "гц",
    "штук": "шт",
    "штука": "шт",
}

# Пробелы-разделители тысяч («2 190» → «2190») и запятая как
# десятичный разделитель («1,5 л» → «1.5 л»).
_THOUSANDS_SPACE = re.compile(r"(?<=\d)[\s\u00a0\u202f](?=\d)")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")


def normalize_value(raw: str) -> str:
    """Привести значение к канонической строке для сравнения.

    NFKC + нижний регистр + единые тире и десятичные разделители +
    снятие пробелов-разделителей тысяч + канонизация синонимов единиц
    («ватт» → «вт»). Результат сравнивается обычным ==.
    """
    text = unicodedata.normalize("NFKC", str(raw)).lower().translate(_DASHES)
    text = text.replace("ё", "е")
    text = _DECIMAL_COMMA.sub(".", text)
    text = _THOUSANDS_SPACE.sub("", text)
    words = [_UNIT_SYNONYMS.get(word, word) for word in text.split()]
    text = re.sub(r"\s+", " ", " ".join(words)).strip()
    return text.strip(" .,;:!?")
