"""Тесты метрик шага 8: нормализация и детерминированные метрики.

LLM и база здесь не нужны: метрики — чистые функции, в тестах —
заготовленные карточки и фрагменты.
"""
from evals.metrics import characteristics_score, citation_probes_score
from evals.normalize import normalize_value


def test_normalize_collapses_case_punctuation_and_units():
    """«800 Вт», «800 вт» и «800 Вт.» — одно значение."""
    assert normalize_value("800 Вт") == normalize_value("800 вт")
    assert normalize_value("800 Вт") == normalize_value("800 Вт.")


def test_normalize_decimal_comma_and_thousands_space():
    """«1,5 л» = «1.5 л», «2 190» = «2190»."""
    assert normalize_value("1,5 л") == normalize_value("1.5 л")
    assert normalize_value("2 190 ₽") == normalize_value("2190 ₽")


def test_normalize_unit_synonyms():
    """Синонимы единиц канонизируются («ватт» → «вт»)."""
    assert normalize_value("800 ватт") == normalize_value("800 Вт")


def test_characteristics_score_counts_normalized_matches():
    """Доля совпавших с эталоном; регистр и пунктуация не мешают."""
    expected = {"Мощность": "800 Вт", "Вес": "1,2 кг"}
    actual = {"мощность": "800 вт.", "Вес": "1.2 кг"}
    assert characteristics_score(expected, actual) == 1.0


def test_characteristics_score_misses_count():
    """Несовпавшее значение и отсутствующий ключ снижают долю."""
    expected = {"Мощность": "800 Вт", "Вес": "1,2 кг", "Цвет": "графитовый"}
    actual = {"Мощность": "1000 Вт", "Цвет": "графитовый"}
    assert characteristics_score(expected, actual) == 1 / 3


def test_characteristics_score_empty_expected_is_perfect():
    """Эталон без характеристик: проверять нечего — 1.0."""
    assert characteristics_score({}, {"Мощность": "800 Вт"}) == 1.0


def _chunks_by_id() -> dict[int, dict]:
    """Фиксированные фрагменты для проб цитирования."""
    return {
        10: {"id": 10, "text": "Мощность 800 Вт, гарантия 24 месяца", "page": 1},
        11: {"id": 11, "text": "Цвет: графитовый", "page": 2},
    }


def test_citation_probes_score_confirms_matching_source():
    """Проба подтверждена: источник карточки содержит текст пробы."""
    probes = {"Мощность": {"page": 1, "text_contains": "800 Вт"}}
    sources = [{"chunk_id": 10, "document": "blender_passport.pdf"}]
    assert citation_probes_score(probes, sources, _chunks_by_id()) == 1.0


def test_citation_probes_score_detects_wrong_page():
    """Страница пробы не совпала — проба не подтверждена."""
    probes = {"Мощность": {"page": 2, "text_contains": "800 Вт"}}
    sources = [{"chunk_id": 10, "document": "blender_passport.pdf"}]
    assert citation_probes_score(probes, sources, _chunks_by_id()) == 0.0


def test_citation_probes_score_detects_unsourced_fact():
    """Факт есть, но карточка сослалась не на тот фрагмент — 0."""
    probes = {"Мощность": {"page": 1, "text_contains": "800 Вт"}}
    sources = [{"chunk_id": 11, "document": "blender_passport.pdf"}]
    assert citation_probes_score(probes, sources, _chunks_by_id()) == 0.0


def test_citation_probes_score_page_null_means_any():
    """page: null в эталоне — страница не проверяется."""
    probes = {"BLD-800": {"page": None, "text_contains": "2 190"}}
    sources = [{"chunk_id": 12, "document": "blender_kp.docx"}]
    chunks = {12: {"id": 12, "text": "BLD-800, цена 2 190 ₽", "page": None}}
    assert citation_probes_score(probes, sources, chunks) == 1.0
