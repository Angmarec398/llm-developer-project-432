"""Расчёт стоимости вызова модели из расхода токенов.

Цены — за миллион токенов, живут в настройках (app/core/config.py).
Деньги не считаем числом с плавающей точкой: считаем в Decimal,
храним в БД как NUMERIC — иначе копейки съезжают на суммах.
"""
from decimal import Decimal

_MILLION = Decimal(1_000_000)


def call_cost(
    prompt_tokens: int,
    completion_tokens: int,
    price_input_per_mtok: Decimal,
    price_output_per_mtok: Decimal,
) -> Decimal:
    """Стоимость одного вызова: входные и выходные токены по своим ценам.

    prompt_tokens/completion_tokens — расход, который модель возвращает
    вместе с ответом; цены — Decimal за миллион токенов. Результат —
    точный Decimal, округлять не нужно: NUMERIC в БД держит все знаки.
    """
    input_cost = Decimal(prompt_tokens) * price_input_per_mtok
    output_cost = Decimal(completion_tokens) * price_output_per_mtok
    return (input_cost + output_cost) / _MILLION
