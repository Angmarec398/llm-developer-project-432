"""Расчёт стоимости: точность Decimal на заданном расходе токенов."""
from decimal import Decimal

from app.llm.cost import call_cost


def test_zero_price_is_free():
    """Цены ноль — вызов бесплатен, как у локальной LM Studio."""
    cost = call_cost(1500, 400, Decimal(0), Decimal(0))
    assert cost == Decimal(0)


def test_known_usage_known_price():
    """Заданный расход × заданные цены = точная стоимость, без float.

    1000 входных по 2 за миллион + 500 выходных по 6 за миллион
    = 0.002 + 0.003 = 0.005.
    """
    cost = call_cost(1000, 500, Decimal(2), Decimal(6))
    assert cost == Decimal("0.005")


def test_no_float_drift():
    """Дробности не накапливают ошибку: 10 мелких вызовов = 1 ровный итог."""
    one_call = call_cost(333, 77, Decimal("0.3"), Decimal("1.1"))
    total = one_call * 10
    direct = call_cost(3330, 770, Decimal("0.3"), Decimal("1.1"))
    assert total == direct
