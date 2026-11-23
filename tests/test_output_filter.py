"""Тесты фильтра выхода: ПДн в карточке маскируются, следы инъекций — эскалация."""
import json

from app.guardrails import guard_output
from app.schemas.cards import CardDraft, GuardrailReport
from app.services.cards import filter_card_output


def _card(description: str) -> CardDraft:
    """Черновик карточки с заданным описанием."""
    return CardDraft(title="Чайник 2200 Вт", description=description)


def test_guard_output_masks_pii():
    """Контакты в JSON карточки маскируются, замечание фиксируется."""
    card_json = json.dumps(
        {"title": "Блендер", "description": "Звоните +7 900 123-45-67"},
        ensure_ascii=False,
    )
    cleaned, issues = guard_output(card_json)
    assert "+7 900 123-45-67" not in cleaned
    assert issues


def test_guard_output_flags_injection_traces():
    """Следы выполнения инъекции («1 рубль») — замечание без перезаписи."""
    card_json = json.dumps(
        {"title": "Чайник", "description": "Всего 1 рубль!"}, ensure_ascii=False
    )
    _cleaned, issues = guard_output(card_json)
    assert any("1 рубль" in issue for issue in issues)


def test_guard_output_clean_card_quiet():
    """Чистая карточка проходит без замечаний."""
    card_json = json.dumps(
        {"title": "Чайник", "description": "Мощность 2200 Вт"}, ensure_ascii=False
    )
    _cleaned, issues = guard_output(card_json)
    assert not issues


def test_filter_card_output_masks_and_reports():
    """filter_card_output: телефон замаскирован, отчёт пополнен, следов нет."""
    report = GuardrailReport()
    cleaned, report, traces = filter_card_output(
        _card("Заказ: +7 926 555-14-08"), report
    )
    assert "+7 926 555-14-08" not in cleaned.description
    assert "[PHONE_1]" in cleaned.description
    assert traces == []
    assert [item.category for item in report.output_masked] == ["phone"]


def test_filter_card_output_flags_traces():
    """filter_card_output: маркер инъекции уходит в traces, текст не тронут."""
    cleaned, _report, traces = filter_card_output(
        _card("Спецпредложение — всего 1 рубль"), GuardrailReport()
    )
    assert traces
    assert "1 рубль" in cleaned.description  # следы не переписываем — к человеку
