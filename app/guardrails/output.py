"""Фильтр выхода (шаг 9): карточка перед отдачей — третий рубеж.

Модель могла выполнить инъекцию частично: первые два рубежа смотрели
на фрагменты на входе, но сгенерированный текст ими не проверялся.
Поэтому финальная карточка прогоняется теми же детекторами:

- контакты, просочившиеся в текст карточки, маскируются метками —
  карточка остаётся чистой и уходит дальше;
- следы служебных инструкций (приказы игнорировать инструкции,
  маркеры ролей) — причина эскалации к человеку: текст уже отравлен,
  замена меткой смысла не вернёт.

Результаты добавляются в отчёт (output_masked) — оператор видит, что
защита сработала и на выходе.
"""
from app.core.logging import get_logger
from app.guardrails import injection
from app.guardrails.pii import PIIFinding, mask_pii, masked_labels
from app.schemas.cards import GuardrailReport, MaskedItem, ProductCard

log = get_logger(__name__)


def filter_card_output(
    card: ProductCard, report: GuardrailReport
) -> tuple[ProductCard, list[str]]:
    """Проверить карточку детекторами входа и очистить её.

    Каждая строка сериализованной карточки маскируется (контакты →
    метки), найденное дописывается в отчёт. Затем текст очищенной
    карточки проверяется правилами инъекций: их срабатывание —
    следы выполненной инъекции. Возвращает (очищенная карточка,
    следы инъекций): непустые следы — эскалация к человеку.
    """
    findings: list[PIIFinding] = []

    def _mask_strings(value):
        """Рекурсивно замаскировать ПДн во всех строках структуры."""
        if isinstance(value, str):
            masked = mask_pii(value)
            findings.extend(masked.findings)
            return masked.text
        if isinstance(value, list):
            return [_mask_strings(item) for item in value]
        if isinstance(value, dict):
            return {
                key: _mask_strings(item) for key, item in value.items()
            }
        return value

    cleaned = ProductCard.model_validate(_mask_strings(card.model_dump()))
    report.output_masked.extend(
        MaskedItem(category=category, label=label)
        for category, label in masked_labels(findings)
    )
    traces = injection.rule_hits(cleaned.model_dump_json())
    if findings or traces:
        log.info(
            "guardrail_output_filter",
            masked=len(findings),
            injection_traces=traces,
        )
    return cleaned, traces
