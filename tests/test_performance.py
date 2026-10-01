import pandas as pd
import pytest

from datetime import datetime
from src.strategies.base import Signal
from src.backtesting.execution import Trade
from src.metrics.performance import (
    total_return,
    periodic_returns,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    trade_returns,
    win_rate,
    profit_factor,
    generate_performance_report,
    buy_and_hold_return,
    number_of_trades,
    average_winning_trade,
    average_losing_trade,
    largest_winning_trade,
    largest_losing_trade,
    gross_profit,
    gross_loss,
    average_holding_period,
)


def test_total_return():

    equity_curve = pd.Series(
        [1000, 1050, 1100, 1200]
    )

    result = total_return(equity_curve)

    assert result == pytest.approx(0.20)


def test_total_return_loss():

    equity_curve = pd.Series(
        [1000, 950, 900]
    )

    result = total_return(equity_curve)

    assert result == pytest.approx(-0.10)


def test_empty_equity_curve():

    equity_curve = pd.Series(
        dtype=float
    )

    with pytest.raises(ValueError):
        total_return(equity_curve)


def test_invalid_initial_value():

    equity_curve = pd.Series(
        [0, 100, 200]
    )

    with pytest.raises(ValueError):
        total_return(equity_curve)

def test_cagr():

    dates = pd.to_datetime([
        "2020-01-01",
        "2021-01-01",
        "2022-01-01"
    ])

    equity_curve = pd.Series(
        [1000, 1100, 1210],
        index=dates
    )

    result = cagr(equity_curve)

    assert result == pytest.approx(0.10, abs=0.001)

def test_cagr_requires_multiple_dates():

    dates = pd.to_datetime([
        "2025-01-01"
    ])

    equity_curve = pd.Series(
        [1000],
        index=dates
    )

    with pytest.raises(ValueError):
        cagr(equity_curve)

def test_periodic_returns():

    equity_curve = pd.Series(
        [1000, 1100, 990]
    )

    returns = periodic_returns(equity_curve)

    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)

def test_periodic_returns_invalid_values():

    equity_curve = pd.Series(
        [1000, 0, 1100]
    )

    with pytest.raises(ValueError):
        periodic_returns(equity_curve)

def test_annualized_volatility():

    equity_curve = pd.Series(
        [100, 101, 99, 102, 100]
    )

    result = annualized_volatility(
        equity_curve,
        periods_per_year=252
    )

    expected_daily_volatility = (
        periodic_returns(equity_curve).std(ddof=1)
    )

    expected = expected_daily_volatility * (252 ** 0.5)

    assert result == pytest.approx(expected)

def test_annualized_volatility_requires_enough_data():

    equity_curve = pd.Series([100])

    with pytest.raises(ValueError):
        annualized_volatility(equity_curve)

def test_sharpe_ratio():

    equity_curve = pd.Series(
        [100, 102, 101, 104, 103, 106]
    )

    result = sharpe_ratio(
        equity_curve,
        risk_free_rate=0.0,
        periods_per_year=252
    )

    returns = periodic_returns(equity_curve)

    expected = (
        returns.mean()
        / returns.std(ddof=1)
    ) * (252 ** 0.5)

    assert result == pytest.approx(expected)

def test_sharpe_ratio_with_risk_free_rate():

    equity_curve = pd.Series(
        [100, 102, 101, 104, 103, 106]
    )

    result = sharpe_ratio(
        equity_curve,
        risk_free_rate=0.05,
        periods_per_year=252
    )

    assert isinstance(result, float)

def test_sharpe_ratio_requires_enough_data():

    equity_curve = pd.Series([100])

    with pytest.raises(ValueError):
        sharpe_ratio(equity_curve)

def test_max_drawdown():

    equity_curve = pd.Series(
        [1000, 1200, 1500, 1300, 1100, 1400]
    )

    result = max_drawdown(equity_curve)

    assert result == pytest.approx(
        -0.2666666667
    )

def test_max_drawdown_no_decline():

    equity_curve = pd.Series(
        [1000, 1100, 1200, 1300]
    )

    result = max_drawdown(equity_curve)

    assert result == pytest.approx(0.0)

def test_max_drawdown_invalid_values():

    equity_curve = pd.Series(
        [1000, 1200, 0, 1300]
    )

    with pytest.raises(ValueError):
        max_drawdown(equity_curve)

def test_trade_returns():

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 5),
            signal=Signal.SELL,
            price=120,
            quantity=10,
            transaction_cost=0
        )
    ]

    returns = trade_returns(trades)

    assert len(returns) == 1
    assert returns[0] == pytest.approx(0.20)

def test_losing_trade():

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 5),
            signal=Signal.SELL,
            price=80,
            quantity=10,
            transaction_cost=0
        )
    ]

    returns = trade_returns(trades)

    assert returns[0] == pytest.approx(-0.20)

def test_win_rate():

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 2),
            signal=Signal.SELL,
            price=120,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 3),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 4),
            signal=Signal.SELL,
            price=90,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 5),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 6),
            signal=Signal.SELL,
            price=110,
            quantity=10,
            transaction_cost=0
        )
    ]

    result = win_rate(trades)

    assert result == pytest.approx(2 / 3)

def test_profit_factor():

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 2),
            signal=Signal.SELL,
            price=120,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 3),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 4),
            signal=Signal.SELL,
            price=90,
            quantity=10,
            transaction_cost=0
        )
    ]

    result = profit_factor(trades)

    # +200 profit / 100 loss
    assert result == pytest.approx(2.0)

def test_profit_factor_no_losses():

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 2),
            signal=Signal.SELL,
            price=120,
            quantity=10,
            transaction_cost=0
        )
    ]

    result = profit_factor(trades)

    assert result == float("inf")

def test_buy_and_hold_return():

    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
                120
            ]
        }
    )

    result = buy_and_hold_return(data)

    assert result == pytest.approx(0.20)

def test_buy_and_hold_missing_close():

    data = pd.DataFrame(
        {
            "Open": [100, 110]
        }
    )

    with pytest.raises(ValueError):
        buy_and_hold_return(data)

def test_buy_and_hold_empty_data():

    data = pd.DataFrame(
        columns=["Close"]
    )

    with pytest.raises(ValueError):
        buy_and_hold_return(data)

def test_trade_level_metrics():

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=1
        ),
        Trade(
            date=datetime(2025, 1, 6),
            signal=Signal.SELL,
            price=120,
            quantity=10,
            transaction_cost=1
        ),

        Trade(
            date=datetime(2025, 2, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=1
        ),
        Trade(
            date=datetime(2025, 2, 4),
            signal=Signal.SELL,
            price=90,
            quantity=10,
            transaction_cost=1
        )
    ]

    # First trade:
    #
    # Buy  = 100 * 10 + 1 = 1001
    # Sell = 120 * 10 - 1 = 1199
    # PnL  = 198
    #
    # Second trade:
    #
    # Buy  = 100 * 10 + 1 = 1001
    # Sell = 90 * 10 - 1 = 899
    # PnL  = -102

    assert number_of_trades(trades) == 2

    assert gross_profit(trades) == pytest.approx(198)

    assert gross_loss(trades) == pytest.approx(102)

    assert average_winning_trade(
        trades
    ) == pytest.approx(198)

    assert average_losing_trade(
        trades
    ) == pytest.approx(-102)

    assert largest_winning_trade(
        trades
    ) == pytest.approx(198)

    assert largest_losing_trade(
        trades
    ) == pytest.approx(-102)

    assert profit_factor(
        trades
    ) == pytest.approx(198 / 102)

    assert average_holding_period(
        trades
    ) == pytest.approx(4.0)

def test_trade_metrics_ignore_open_position():

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        )
    ]

    assert number_of_trades(trades) == 0
    assert win_rate(trades) == 0.0
    assert gross_profit(trades) == 0.0
    assert gross_loss(trades) == 0.0
    assert profit_factor(trades) == 0.0
    assert average_winning_trade(trades) == 0.0
    assert average_losing_trade(trades) == 0.0
    assert largest_winning_trade(trades) == 0.0
    assert largest_losing_trade(trades) == 0.0
    assert average_holding_period(trades) == 0.0

def test_generate_performance_report():

    history = pd.DataFrame(
        {
            "total_value": [
                1000,
                1050,
                1020,
                1100
            ]
        },
        index=pd.to_datetime([
            "2025-01-01",
            "2025-06-01",
            "2025-09-01",
            "2026-01-01"
        ])
    )

    trades = [
        Trade(
            date=datetime(2025, 1, 1),
            signal=Signal.BUY,
            price=100,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 6),
            signal=Signal.SELL,
            price=120,
            quantity=10,
            transaction_cost=0
        )
    ]

    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
                120
            ]
        },
        index=pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03"
        ])
    )

    report = generate_performance_report(
        history,
        trades,
        data
    )

    # --------------------------------------------------
    # Check that all expected metrics exist
    # --------------------------------------------------

    assert "total_return" in report
    assert "cagr" in report
    assert "annualized_volatility" in report
    assert "sharpe_ratio" in report
    assert "max_drawdown" in report

    assert "number_of_trades" in report
    assert "win_rate" in report
    assert "profit_factor" in report

    assert "gross_profit" in report
    assert "gross_loss" in report

    assert "average_winning_trade" in report
    assert "average_losing_trade" in report

    assert "largest_winning_trade" in report
    assert "largest_losing_trade" in report

    assert "average_holding_period_days" in report

    assert "buy_and_hold_return" in report

    # --------------------------------------------------
    # Check values
    # --------------------------------------------------

    assert report["total_return"] == pytest.approx(0.10)

    assert report["number_of_trades"] == 1

    assert report["win_rate"] == pytest.approx(1.0)

    assert report["gross_profit"] == pytest.approx(200)

    assert report["gross_loss"] == pytest.approx(0.0)

    assert report["average_winning_trade"] == pytest.approx(200)

    assert report["average_losing_trade"] == pytest.approx(0.0)

    assert report["largest_winning_trade"] == pytest.approx(200)

    assert report["largest_losing_trade"] == pytest.approx(0.0)

    assert report["average_holding_period_days"] == pytest.approx(5.0)

    assert report["buy_and_hold_return"] == pytest.approx(0.20)