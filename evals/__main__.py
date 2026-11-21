"""CLI прогона метрик: python -m evals [--full].

По умолчанию — быстрое подмножество eval_defaults; --full гоняет
весь эталонный набор (локальная модель медленная, полный прогон —
когда есть время).
"""
import argparse
import asyncio

from app.core.db import close_pool, init_pool
from app.core.logging import setup_logging
from app.llm.client import setup_llm
from evals.runner import run_eval


def main() -> None:
    """Разобрать аргументы и запустить прогон."""
    parser = argparse.ArgumentParser(description="Метрики качества карточек")
    parser.add_argument(
        "--full",
        action="store_true",
        help="весь эталонный набор (иначе — eval_defaults)",
    )
    arguments = parser.parse_args()
    setup_logging()
    init_pool()
    setup_llm()
    try:
        asyncio.run(run_eval(full=arguments.full))
    finally:
        close_pool()


if __name__ == "__main__":
    main()
