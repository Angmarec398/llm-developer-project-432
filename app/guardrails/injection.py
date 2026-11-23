"""Детектор prompt injection: regex-эвристики + мини-LLM на дешёвой модели.

Документ поставщика — untrusted-контекст: инъекция прячется в мелком шрифте
PDF. Regex ловит грубое, LLM — завуалированное (и только по подозрительным
чанкам — дешевле, чем гнать все через модель).

LLM-детектор без output_type: грамматика structured output LM Studio
обнуляет свободные поля (ADR-0004) — разбор ответа через parse_llm_json.
"""
import re
from dataclasses import dataclass

from app.core.logging import get_logger

log = get_logger(__name__)

# Паттерны грубых инъекций (RU/EN)
_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"игнорируй\s+(все\s+)?предыдущ",
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?(previous|above)",
        r"\bsystem\s*:",                      # попытка подменить роль
        r"<\|im_start\|>|<\|system\|>",        # маркеры чат-шаблонов
        r"reveal\s+your\s+(prompt|instructions)",
        r"ты\s+теперь\s+(должен|обязан)",
        r"нов[ыые]\s+инструкции:",
        r"укажи\s+цену\s+\d",                 # бизнес-аномалия из датасета
    )
]


@dataclass
class InjectionVerdict:
    """Вердикт детектора: подозрителен ли фрагмент и почему."""
    suspicious: bool
    reason: str = ""


def detect_injection_regex(text: str) -> InjectionVerdict:
    """Быстрый детектор: любой паттерн → suspicious с причиной."""
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return InjectionVerdict(True, reason=f"regex: {match.group(0)!r}")
    return InjectionVerdict(False)


async def detect_injection_llm(text: str) -> InjectionVerdict:
    """Мини-LLM детектор: узкий промпт, служебная модель. Для подозрительных чанков."""
    from app.agents.prompts import injection_screen_agent
    from app.llm.client import run_agent
    from app.services.json_utils import parse_llm_json

    out = await run_agent(injection_screen_agent, "ФРАГМЕНТ:\n" + text[:4000])
    data = parse_llm_json(out if isinstance(out, str) else str(out))
    return InjectionVerdict(bool(data.get("injection")), str(data.get("reason", "")))
