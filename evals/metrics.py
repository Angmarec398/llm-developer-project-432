"""Метрики качества карточки против эталона (шаг 8).

Две метрики детерминированные и считаются здесь:
- characteristics_score — доля эталонных характеристик, совпавших
  с карточкой (значения сравниваются нормализованные);
- citation_probes_score — доля проб цитирования, где заявленный
  карточкой фрагмент действительно содержит то, на что ссылаются.

Третья (faithfulness, модель-судья) — этап 6. Функции чистые:
работают с dict/str, БД и LLM не трогают — их вызывает раннер
(evals/runner.py), который и подставляется в тестах.
"""
from app.schemas.cards import FaithfulnessVerdict
from evals.normalize import normalize_value


def characteristics_score(
    expected: dict[str, str], actual: dict[str, str]
) -> float:
    """Доля эталонных характеристик, совпавших с карточкой.

    Ключ ищется в нормализованном виде, значение сравнивается
    нормализованное. Эталон пуст — 1.0: проверять нечего.
    """
    if not expected:
        return 1.0
    normalized_actual = {
        normalize_value(key): normalize_value(value)
        for key, value in actual.items()
    }
    matched = sum(
        1
        for key, value in expected.items()
        if normalized_actual.get(normalize_value(key)) == normalize_value(value)
    )
    return matched / len(expected)


def citation_probes_score(
    probes: dict[str, dict],
    sources: list[dict],
    chunks_by_id: dict[int, dict],
) -> float:
    """Доля проб цитирования, подтверждённых заявленными источниками.

    Для каждой пробы (имя характеристики из эталона) ищем среди
    источников карточки фрагмент, чей текст содержит text_contains
    (нормализованно); если в пробе задана страница — она должна
    совпадать со страницей фрагмента. Не подтверждённая проба —
    либо карточка не сослалась на нужный фрагмент, либо сослалась
    не туда («характеристика верная, но взята не оттуда»).
    Проб нет — 1.0.
    """
    if not probes:
        return 1.0
    confirmed = 0
    for probe in probes.values():
        needle = normalize_value(probe["text_contains"])
        for source in sources:
            chunk = chunks_by_id.get(source["chunk_id"])
            if chunk is None:
                continue
            if probe.get("page") is not None and chunk.get("page") != probe["page"]:
                continue
            if needle in normalize_value(chunk["text"]):
                confirmed += 1
                break
    return confirmed / len(probes)

def faithfulness_score(verdict: FaithfulnessVerdict) -> float:
    """Доля утверждений карточки, подтверждённых источниками.

    Считается из разметки судьи: чем больше unsupported, тем ниже
    метрика. Вердикт пуст — 1.0: проверять нечего. Функция чистая —
    судью вызывает раннер, сюда приходит готовый вердикт.
    """
    supported = len(verdict.supported)
    unsupported = len(verdict.unsupported)
    total = supported + unsupported
    if total == 0:
        return 1.0
    return supported / total
