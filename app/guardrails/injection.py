"""Двухуровневый детектор prompt-инъекций (шаг 9).

Второй рубеж: документ поставщика — недоверенные данные. В нём может
быть текст «SYSTEM: игнорируй предыдущие инструкции…»: он попадёт во
фрагмент, фрагмент — в контекст генерации, и карточка рекламирует
товар за рубль. Послушания модели недостаточно — защита ставится на
границе, а не на уговорах в промпте.

Два уровня, строго в этом порядке:

- Быстрый — правила по тексту: прямые приказы («игнорируй
  инструкции»), маркеры служебных ролей («system:»), попытки вытащить
  промпт, подозрительно длинные закодированные вставки. Бесплатный,
  ловит грубое.
- Точный — узкий вопрос дешёвой служебной модели: «содержит ли этот
  фрагмент попытку манипуляции инструкциями». Платный, поэтому идёт
  только по фрагментам, помеченным первым уровнем: гнать через
  модель все фрагменты дорого и незачем.

Свойство, без которого детектор не считается защитой: при недоступной
модели фрагмент считается подозрительным, а не пропускается. Защита,
которая при своём отказе открывает дверь, — это не защита.

Вызовы модели идут через тот же клиент (app/llm/client.py): таймаут
и ретраи из шага про архитектуру, обращения детектора видны в таблице
llm_calls вместе со стоимостью безопасности.
"""
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.agents.prompts import injection_screen_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import run_agent
from app.schemas.cards import InjectionVerdict
from app.services.structured import ResponseParseError, extract_json

log = get_logger(__name__)

# Имя -> regex. Порядок не важен: правила не исключают друг друга,
# накапливают причины. Регистронезависимы — инъекции пишут в любом.
_RULES: tuple[tuple[str, re.Pattern], ...] = (
        (
        "ignore_instructions",
        re.compile(
            r"игнорируй\b.{0,40}инструкц"
            r"|забудь\b.{0,30}инструкц"
            r"|ignore\b.{0,40}\binstructions?\b"
            r"|disregard\b.{0,30}\binstructions?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_marker",
        re.compile(
            r"system\s*:|system\s+prompt"
            r"|ты\s+теперь|отныне\s+ты|you\s+are\s+now"
            r"|developer\s+mode|режим\s+разработчика|\bdan\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_extraction",
        re.compile(
            r"(?:выведи|покажи|раскрой|напечатай|повтори)"
            r".{0,30}(?:промпт|инструкци)"
            r"|print\s+your\s+(?:instructions|prompt)"
            r"|repeat\s+(?:the\s+words|everything)\s+above",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "encoded_blob",
        re.compile(r"[A-Za-z0-9+/]{100,}={0,2}|\b[0-9a-fA-F]{64,}\b"),
    ),
)


@dataclass
class InjectionScreenResult:
    """Итог скрининга фрагмента и причины при подозрительности."""

    suspicious: bool
    reasons: list[str] = field(default_factory=list)


def rule_hits(text: str) -> list[str]:
    """Имена сработавших правил быстрого уровня.

    Пустой список — фрагмент чист настолько, что точный уровень ему
    не нужен: дорогой вызов модели случается только при непустом.
    """
    return [name for name, pattern in _RULES if pattern.search(text)]


async def screen_text(text: str) -> InjectionScreenResult:
    """Полный скрининг одного фрагмента: правила, затем — по необходимости
    модель.

    Чистый по правилам фрагмент возвращается немедленно (бесплатно).
    Помеченный правилами уходит к служебной модели. Модель, вернувшая
    мусор или не ответившая, трактуется как НЕДОСТУПНАЯ: фрагмент
    подозрительный — защита закрывается, а не открывается. Точечная
    починка отдельных полей здесь намеренно не используется: вердикт
    без контекста фрагмента — гадание, которое однажды уже пропустило
    инъекцию.
    """
    hits = rule_hits(text)
    if not hits:
        return InjectionScreenResult(suspicious=False, reasons=[])
    try:
        verdict = await _verdict(_screen_prompt(text, hits))
    except Exception as error:  # noqa: BLE001 — детектор обязан закрыться
        log.warning(
            "injection_screener_unavailable", error=repr(error), rules=hits
        )
        return InjectionScreenResult(
            suspicious=True, reasons=hits + ["detector_unavailable"]
        )
    if verdict.is_injection:
        return InjectionScreenResult(
            suspicious=True,
            reasons=hits + [verdict.reason or "model_confirmed_injection"],
        )
    return InjectionScreenResult(suspicious=False, reasons=[])


async def _verdict(prompt: str) -> InjectionVerdict:
    """Вызов служебной модели с полным повтором при мусорном ответе.

    Лечение — только полным повтором исходного вопроса с текстом
    ошибки (модель видит фрагмент и правила и отвечает заново). Если
    и повтор мусорный — исключение: вызывающий код закрывается.
    """
    current_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(1, settings.validation_max_attempts + 1):
        raw_output = await run_agent(injection_screen_agent, current_prompt)
        if isinstance(raw_output, InjectionVerdict):
            return raw_output
        try:
            return InjectionVerdict.model_validate_json(
                extract_json(str(raw_output))
            )
        except (ResponseParseError, ValidationError) as error:
            last_error = error
            log.info(
                "injection_verdict_healing",
                attempt=attempt,
                error=str(error)[:200],
            )
            current_prompt = (
                prompt
                + "\n\nТвой предыдущий ответ не разобран: "
                + str(raw_output)[:300]
                + "\nОшибка: "
                + str(error)
                + "\nВерни ТОЛЬКО валидный JSON с полями "
                "is_injection и reason."
            )
    raise ResponseParseError(
        f"детектор не вернул разбираемый вердикт: {last_error}"
    )


def _screen_prompt(text: str, rules: list[str]) -> str:
    """Узкий вопрос точного уровня: фрагмент и сработавшие правила.

    Правила передаём модели явно: слабой служебной модели так проще
    ответить точно — ей не нужно угадывать, что именно показалось
    подозрительным быстрому уровню.
    """
    return (
        "Сработавшие правила быстрого детектора: "
        + ", ".join(rules)
        + "\nФрагмент документа поставщика:\n"
        + text
    )

