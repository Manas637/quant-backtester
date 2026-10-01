from datetime import datetime

import pytest

from src.backtesting.execution import ExecutionEngine
from src.strategies.base import Signal


def test_buy_slippage():

    engine = ExecutionEngine(
        transaction_cost=0,
        slippage=0.001
    )

    trade = engine.execute(
        date=datetime(2025, 1, 1),
        signal=Signal.BUY,
        price=100,
        quantity=10
    )

    assert trade.price == pytest.approx(100.10)

def test_sell_slippage():

    engine = ExecutionEngine(
        transaction_cost=0,
        slippage=0.001
    )

    trade = engine.execute(
        date=datetime(2025, 1, 1),
        signal=Signal.SELL,
        price=100,
        quantity=10
    )

    assert trade.price == pytest.approx(99.90)

def test_zero_slippage():

    engine = ExecutionEngine(
        transaction_cost=0,
        slippage=0
    )

    trade = engine.execute(
        date=datetime(2025, 1, 1),
        signal=Signal.BUY,
        price=100,
        quantity=10
    )

    assert trade.price == pytest.approx(100)

def test_negative_slippage():

    with pytest.raises(ValueError):
        ExecutionEngine(
            transaction_cost=0,
            slippage=-0.001
        )

def test_transaction_cost_uses_slipped_price():

    engine = ExecutionEngine(
        transaction_cost=0.01,
        slippage=0.01
    )

    trade = engine.execute(
        date=datetime(2025, 1, 1),
        signal=Signal.BUY,
        price=100,
        quantity=10
    )

    assert trade.price == pytest.approx(101)

    assert trade.transaction_cost == pytest.approx(
        10.10
    )