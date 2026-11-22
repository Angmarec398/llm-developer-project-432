"""Тесты детекторов ПДн (шаг 9): маскирование, контрольные суммы, ложные срабатывания."""
from app.guardrails.pii import mask_pii


def test_phone_replaced_with_label():
    """Телефон из эталона уходит, на его месте — метка."""
    result = mask_pii("Контакты менеджера: +7 926 555-14-08, звоните.")
    assert "+7 926 555-14-08" not in result.text
    assert "телефон номер один" in result.text
    assert [finding.category for finding in result.findings] == ["phone"]


def test_phone_format_with_brackets_and_8():
    """Другие российские записи того же номера тоже ловятся."""
    result = mask_pii("Тел.: 8 (926) 555-14-08 или +7(926)5551408")
    assert "8 (926) 555-14-08" not in result.text
    assert "+7(926)5551408" not in result.text
    assert result.text.count("телефон") == 2


def test_email_masked():
    """Почта менеджера из эталона уходит, метка на месте."""
    result = mask_pii("Пишите: a.smirnova@technodom.example — отвечу.")
    assert "a.smirnova@technodom.example" not in result.text
    assert "электронная почта номер один" in result.text


def test_taxpayer_number_with_valid_checksum_masked():
    """ИНН из эталона (валидная контрольная сумма) маскируется."""
    result = mask_pii("Наш ИНН 7712345671, оплатите счёт.")
    assert "7712345671" not in result.text
    assert "номер налогоплательщика номер один" in result.text


def test_taxpayer_number_with_bad_checksum_untouched():
    """Похожая последовательность с битой суммой — НЕ трогаем."""
    result = mask_pii("Артикул 7712345678 из каталога.")
    assert result.text == "Артикул 7712345678 из каталога."
    assert result.findings == []


def test_card_number_with_luhn_masked():
    """Номер карты с валидным Луна маскируется."""
    result = mask_pii("Оплата картой 4000 0000 0000 0002.")
    assert "4000 0000 0000 0002" not in result.text
    assert "номер карты номер один" in result.text


def test_card_number_with_bad_luhn_untouched():
    """Битый по Луна номер — не находка."""
    result = mask_pii("Счёт 4000 0000 0000 0003 в документе.")
    assert result.findings == []


def test_plain_text_untouched():
    """Чистый текст без ПДн проходит без изменений."""
    source = "Чайник электрический, мощность 2200 Вт, объём 1.7 л."
    result = mask_pii(source)
    assert result.text == source
    assert result.findings == []


def test_article_sum_date_not_masked():
    """Артикулы, суммы, даты не маскируются — ИНН-детектор их не берёт."""
    result = mask_pii(
        "Артикул BLD-800, цена 2 190 ₽, отгрузка 12.05.2026, поставка 500 шт."
    )
    assert result.findings == []
    assert "BLD-800" in result.text
    assert "12.05.2026" in result.text


def test_several_phones_numbered():
    """Несколько находок одной категории нумеруются словами по порядку."""
    result = mask_pii("Директор +7 926 555-14-08, бухгалтерия +7 495 111-22-33.")
    assert "телефон номер один" in result.text
    assert "телефон номер два" in result.text
    assert "+7" not in result.text


def test_findings_do_not_store_raw_values():
    """Находка помнит только категорию и позицию — сырого значения нет."""
    result = mask_pii("Телефон +7 926 555-14-08.")
    finding = result.findings[0]
    assert finding.category == "phone"
    assert not hasattr(finding, "raw")
    assert len(vars(finding)) == 3  # category, start, end
