import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import pandas as pd
import pytest

from datetime import datetime

from src.backtesting.execution import Trade
from src.strategies.base import Signal

from src.visualization.plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_price_with_trades,
    plot_strategy_vs_benchmark
)

def test_plot_equity_curve():

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
            "2025-01-02",
            "2025-01-03",
            "2025-01-04"
        ])
    )

    fig, ax = plot_equity_curve(
        history,
        show=False
    )

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1

def test_plot_equity_curve_missing_column():

    history = pd.DataFrame(
        {
            "cash": [1000, 900]
        }
    )

    with pytest.raises(ValueError):
        plot_equity_curve(
            history,
            show=False
        )


def test_plot_equity_curve_empty_history():

    history = pd.DataFrame(
        columns=["total_value"]
    )

    with pytest.raises(ValueError):
        plot_equity_curve(
            history,
            show=False
        )

def test_plot_drawdown():

    history = pd.DataFrame(
        {
            "total_value": [
                1000,
                1200,
                1100,
                900,
                1000
            ]
        },
        index=pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05"
        ])
    )

    fig, ax = plot_drawdown(
        history,
        show=False
    )

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1

def test_plot_drawdown_missing_column():

    history = pd.DataFrame(
        {
            "cash": [1000, 900]
        }
    )

    with pytest.raises(ValueError):
        plot_drawdown(
            history,
            show=False
        )


def test_plot_drawdown_empty_history():

    history = pd.DataFrame(
        columns=["total_value"]
    )

    with pytest.raises(ValueError):
        plot_drawdown(
            history,
            show=False
        )

def test_plot_price_with_trades():

    data = pd.DataFrame(
        {
            "Close": [
                100,
                105,
                110,
                105,
                95
            ]
        },
        index=pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05"
        ])
    )

    trades = [
        Trade(
            date=datetime(2025, 1, 2),
            signal=Signal.BUY,
            price=105,
            quantity=10,
            transaction_cost=0
        ),
        Trade(
            date=datetime(2025, 1, 5),
            signal=Signal.SELL,
            price=95,
            quantity=10,
            transaction_cost=0
        )
    ]

    fig, ax = plot_price_with_trades(
        data,
        trades,
        show=False
    )

    assert fig is not None
    assert ax is not None

    # One line for Close price
    assert len(ax.lines) == 1

    # Two trade markers
    assert len(ax.collections) == 2
    legend = ax.get_legend()

    labels = [
        text.get_text()
        for text in legend.get_texts()
    ]

    assert labels == ["Close Price", "BUY", "SELL"]

def test_plot_price_with_trades_missing_close():

    data = pd.DataFrame(
        {
            "Open": [100, 105]
        }
    )

    with pytest.raises(ValueError):
        plot_price_with_trades(
            data,
            [],
            show=False
        )


def test_plot_price_with_trades_empty_data():

    data = pd.DataFrame(
        columns=["Close"]
    )

    with pytest.raises(ValueError):
        plot_price_with_trades(
            data,
            [],
            show=False
        )

def test_plot_strategy_vs_benchmark():

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
            "2025-01-02",
            "2025-01-03",
            "2025-01-04"
        ])
    )

    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
                90,
                120
            ]
        },
        index=pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04"
        ])
    )

    fig, ax = plot_strategy_vs_benchmark(
        history,
        data,
        show=False
    )

    assert fig is not None
    assert ax is not None

    # Strategy + benchmark
    assert len(ax.lines) == 2

def test_plot_strategy_vs_benchmark_missing_total_value():

    history = pd.DataFrame(
        {
            "cash": [1000, 1100]
        }
    )

    data = pd.DataFrame(
        {
            "Close": [100, 110]
        }
    )

    with pytest.raises(ValueError):
        plot_strategy_vs_benchmark(
            history,
            data,
            show=False
        )


def test_plot_strategy_vs_benchmark_missing_close():

    history = pd.DataFrame(
        {
            "total_value": [1000, 1100]
        }
    )

    data = pd.DataFrame(
        {
            "Open": [100, 110]
        }
    )

    with pytest.raises(ValueError):
        plot_strategy_vs_benchmark(
            history,
            data,
            show=False
        )

def test_plot_strategy_vs_benchmark_uses_overlapping_dates():

    history = pd.DataFrame(
        {
            "total_value": [
                100000,
                105000,
                110000
            ]
        },
        index=pd.to_datetime([
            "2024-01-02",
            "2024-01-03",
            "2024-01-04"
        ])
    )

    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
                120,
                130
            ]
        },
        index=pd.to_datetime([
            "2023-12-28",
            "2023-12-29",
            "2024-01-02",
            "2024-01-03"
        ])
    )

    fig, ax = plot_strategy_vs_benchmark(
        history,
        data,
        show=False
    )

    lines = ax.get_lines()

    assert len(lines) == 2

    strategy_line = lines[0]
    benchmark_line = lines[1]

    # Both lines should begin on the evaluation period.
    assert len(strategy_line.get_xdata()) == 3
    assert len(benchmark_line.get_xdata()) == 2

    assert benchmark_line.get_ydata()[0] == pytest.approx(
        100000
    )

    plt.close(fig)