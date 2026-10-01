import pytest

from src.backtesting.portfolio import Portfolio
from src.backtesting.execution import Trade
from src.strategies.base import Signal

def test_initial_portfolio():

    portfolio = Portfolio(
        initial_capital=100000
    )

    assert portfolio.cash == 100000
    assert portfolio.quantity == 0
    assert portfolio.total_value(100) == 100000

def test_buy():

    portfolio = Portfolio(
        initial_capital=100000
    )

    trade = Trade(
        date="2025-01-01",
        signal=Signal.BUY,
        price=100,
        quantity=10,
        transaction_cost=10
    )

    portfolio.apply_trade(trade)

    assert portfolio.quantity == 10
    assert portfolio.cash == 98990

def test_sell():

    portfolio = Portfolio(
        initial_capital=100000
    )

    buy_trade = Trade(
        date="2025-01-01",
        signal=Signal.BUY,
        price=100,
        quantity=10,
        transaction_cost=10
    )

    portfolio.apply_trade(buy_trade)

    sell_trade = Trade(
        date="2025-01-02",
        signal=Signal.SELL,
        price=110,
        quantity=10,
        transaction_cost=11
    )

    portfolio.apply_trade(sell_trade)

    assert portfolio.quantity == 0
    assert portfolio.cash == 100079

def test_cannot_buy_without_cash():

    portfolio = Portfolio(
        initial_capital=1000
    )

    trade = Trade(
        date="2025-01-01",
        signal=Signal.BUY,
        price=100,
        quantity=20,
        transaction_cost=10
    )

    with pytest.raises(ValueError):
        portfolio.apply_trade(trade)

def test_cannot_sell_without_position():

    portfolio = Portfolio(
        initial_capital=100000
    )

    trade = Trade(
        date="2025-01-01",
        signal=Signal.SELL,
        price=100,
        quantity=10,
        transaction_cost=10
    )

    with pytest.raises(ValueError):
        portfolio.apply_trade(trade)