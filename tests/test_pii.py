"""Тесты маскирования ПДн: метки [KIND_N], ИНН по контрольной сумме,
сырые значения из модуля не возвращаются."""
from app.guardrails.pii import _inn_checksum, contains_pii, mask_pii


def test_phone_masked_with_label():
    """Телефон заменяется меткой [PHONE_1], сырого номера нет."""
    cleaned, entities = mask_pii("Менеджер: +7 926 555-14-08, звоните")
    assert "+7 926 555-14-08" not in cleaned
    assert "[PHONE_1]" in cleaned
    assert entities[0].kind == "phone"
    assert entities[0].placeholder == "[PHONE_1]"


def test_phone_format_with_8():
    """Номер через восьмёрку тоже маскируется."""
    cleaned, _ = mask_pii("тел. 8 900 123 45 67")
    assert "8 900 123 45 67" not in cleaned


def test_email_masked():
    """Почта заменяется меткой [EMAIL_1]."""
    cleaned, entities = mask_pii("пишите на a.smirnova@technodom.example")
    assert "a.smirnova@technodom.example" not in cleaned
    assert "[EMAIL_1]" in cleaned
    assert entities[0].kind == "email"


def test_inn_with_valid_checksum_masked():
    """ИНН с верной контрольной суммой маскируется."""
    assert _inn_checksum("7712345671")
    cleaned, entities = mask_pii("ИНН компании: 7712345671")
    assert "7712345671" not in cleaned
    assert any(entity.kind == "inn" for entity in entities)


def test_inn_with_bad_checksum_untouched():
    """Десять случайных цифр с неверной суммой — не ИНН, не трогаем."""
    assert not _inn_checksum("1234567890")
    cleaned, entities = mask_pii("заказ 1234567890 оформлен")
    assert "1234567890" in cleaned
    assert all(entity.kind != "inn" for entity in entities)


def test_card_number_masked():
    """Номер карты (16 цифр группами) маскируется меткой [CARD_1]."""
    cleaned, entities = mask_pii("карта 4276 5500 1234 5678")
    assert "4276 5500 1234 5678" not in cleaned
    assert any(entity.kind == "card" for entity in entities)


def test_plain_text_untouched():
    """Текст без ПДн возвращается как есть, сущностей нет."""
    text = "Блендер МиксерПро 800, мощность 800 Вт"
    cleaned, entities = mask_pii(text)
    assert cleaned == text
    assert entities == []


def test_articul_and_date_not_masked():
    """Артикулы и даты не принимаются за ПДн."""
    text = "Артикул BLD-800, поставка 12.05.2026"
    cleaned, entities = mask_pii(text)
    assert cleaned == text
    assert entities == []


def test_several_phones_numbered():
    """Несколько телефонов нумеруются по порядку."""
    cleaned, entities = mask_pii(
        "первый +7 926 555-14-08, второй +7 916 111-22-33"
    )
    assert "[PHONE_1]" in cleaned
    assert "[PHONE_2]" in cleaned
    assert len(entities) == 2


def test_findings_do_not_store_raw_values():
    """Сырое значение наружу не возвращается — только категория и метка."""
    _, entities = mask_pii("звоните +7 926 555-14-08")
    assert entities[0].original == ""


def test_contains_pii():
    """Быстрая проверка для фильтра выхода."""
    assert contains_pii("звони +7 926 555-14-08")
    assert not contains_pii("мощность 800 Вт")
