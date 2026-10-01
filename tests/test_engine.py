import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.strategies.moving_average import MovingAverageCrossover
from src.strategies.base import Signal
from src.backtesting.position_sizing import (
    VolatilityTargetSizer
)


def test_backtest_engine_runs():

    data = pd.DataFrame(
        {
            "Open": [
                9,
                10,
                10,
                19,
                21,
                31,
                39,
                21
            ],
            "High": [
                11,
                11,
                11,
                21,
                31,
                41,
                41,
                22
            ],
            "Low": [
                8,
                9,
                9,
                18,
                20,
                30,
                29,
                19
            ],
            "Close": [
                10,
                10,
                10,
                20,
                30,
                40,
                30,
                20
            ],
            "Volume": [
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000
            ]
        },
        index=pd.date_range(
            "2025-01-01",
            periods=8
        )
    )

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3
    )

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0
    )

    history, trades = engine.run(
        data,
        strategy
    )

    assert len(history) == len(data)

    assert "cash" in history.columns
    assert "quantity" in history.columns
    assert "position_value" in history.columns
    assert "total_value" in history.columns

    assert len(trades) > 0

def test_backtest_keeps_position_open_by_default():

    data = pd.DataFrame(
        {
            "Open": [9, 10, 10, 19, 21, 31, 39, 21],
            "High": [11, 11, 11, 21, 31, 41, 41, 22],
            "Low": [8, 9, 9, 18, 20, 30, 29, 19],
            "Close": [10, 10, 10, 20, 30, 40, 30, 20],
            "Volume": [1000] * 8
        },
        index=pd.date_range(
            "2025-01-01",
            periods=8
        )
    )

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3
    )

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0
    )

    history, trades = engine.run(
        data,
        strategy
    )

    assert history.iloc[-1]["quantity"] == 45

def test_backtest_closes_final_position():

    data = pd.DataFrame(
        {
            "Open": [9, 10, 10, 19, 21, 31, 39, 21],
            "High": [11, 11, 11, 21, 31, 41, 41, 22],
            "Low": [8, 9, 9, 18, 20, 30, 29, 19],
            "Close": [10, 10, 10, 20, 30, 40, 30, 20],
            "Volume": [1000] * 8
        },
        index=pd.date_range(
            "2025-01-01",
            periods=8
        )
    )

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3
    )

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0,
        close_positions=True
    )

    history, trades = engine.run(
        data,
        strategy
    )

    assert trades[-1].signal == Signal.SELL
    assert trades[-1].quantity == 45

def test_close_positions_liquidates_final_position():
    data = pd.DataFrame(
        {
            "Open": [10, 10, 20, 20],
            "High": [11, 11, 21, 21],
            "Low": [9, 9, 19, 19],
            "Close": [10, 10, 20, 20],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=4
        )
    )

    strategy = MovingAverageCrossover(
        short_window=1,
        long_window=2
    )

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0,
        position_fraction=0.95,
        close_positions=True,
        slippage=0
    )

    history, trades = engine.run(
        data,
        strategy
    )

    # Final portfolio must be completely liquidated
    assert history.iloc[-1]["quantity"] == 0
    assert history.iloc[-1]["position_value"] == 0

    # Final trade should be a SELL
    assert trades[-1].signal == Signal.SELL

def test_run_with_start_date_uses_warmup_signals():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")

    data = pd.DataFrame(
        {
            "Open":  [10, 10, 10, 20, 30, 40],
            "High":  [11, 11, 11, 21, 31, 41],
            "Low":   [9, 9, 9, 19, 29, 39],
            "Close": [10, 10, 10, 20, 30, 40],
            "Volume": [1000] * 6,
        },
        index=dates,
    )

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3,
    )

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0,
        position_fraction=1.0,
        close_positions=False,
    )

    history, trades = engine.run(
        data,
        strategy,
        start_date=dates[4],
    )

    # Only test-period dates should appear in history
    assert len(history) == 2
    assert history.index[0] == dates[4]
    assert history.index[-1] == dates[5]

    # The BUY signal was generated during the warm-up period
    # and executed on the first test date.
    assert len(trades) == 1
    assert trades[0].date == dates[4]
    assert trades[0].signal == Signal.BUY

    # Portfolio started flat but was allowed to use the warm-up signal.
    assert history.iloc[0]["quantity"] > 0

def test_engine_supports_volatility_target_sizer():

    dates = pd.date_range(
        "2024-01-01",
        periods=30,
        freq="D"
    )

    prices = [
        100 + i
        for i in range(30)
    ]

    data = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1000] * 30,
        },
        index=dates
    )

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3
    )

    sizer = VolatilityTargetSizer(
        target_volatility=0.15,
        max_fraction=1.0
    )

    engine = BacktestEngine(
        initial_capital=100000,
        transaction_cost=0,
        close_positions=True,
        position_sizer=sizer,
        volatility_window=5
    )

    history, trades = engine.run(
        data,
        strategy
    )

    assert not history.empty

    # Every executed trade should have a positive quantity.
    for trade in trades:
        assert trade.quantity > 0

def test_fixed_fraction_sizing_accounts_for_slippage():
    data = pd.DataFrame(
        {
            "Open": [99.0, 99.0, 99.0, 99.0],
            "High": [101.0, 101.0, 101.0, 101.0],
            "Low": [98.0, 98.0, 98.0, 98.0],
            "Close": [99.0, 99.0, 99.0, 99.0],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=4,
            freq="D"
        )
    )

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0.0,
        position_fraction=1.0,
        slippage=0.02
    )

    # A custom strategy that produces BUY on the first day.
    class BuyStrategy:
        def generate_signals(self, data):
            signals = pd.Series(
                Signal.HOLD,
                index=data.index,
                dtype=object
            )
            signals.iloc[0] = Signal.BUY
            return signals

    history, trades = engine.run(
        data,
        BuyStrategy()
    )

    # Raw price = 99
    # BUY execution price = 99 * 1.02 = 100.98
    #
    # 10 shares would cost:
    # 10 * 100.98 = 1009.80
    #
    # Therefore only 9 shares can be purchased.
    assert trades[0].quantity == 9

    # Portfolio must never become negative.
    assert history["cash"].min() >= 0

def test_engine_never_overspends_due_to_buy_slippage():
    data = pd.DataFrame(
        {
            "Open": [99.0, 99.0, 99.0, 99.0],
            "High": [101.0, 101.0, 101.0, 101.0],
            "Low": [98.0, 98.0, 98.0, 98.0],
            "Close": [99.0, 99.0, 99.0, 99.0],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=4,
            freq="D"
        )
    )

    class BuyStrategy:
        def generate_signals(self, data):

            signals = pd.Series(
                Signal.HOLD,
                index=data.index,
                dtype=object
            )

            signals.iloc[0] = Signal.BUY

            return signals

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0.001,
        position_fraction=1.0,
        slippage=0.02
    )

    history, trades = engine.run(
        data,
        BuyStrategy()
    )

    assert len(trades) >= 1

    assert all(
        history["cash"] >= 0
    )

def test_volatility_target_sizing_accounts_for_slippage():
    data = pd.DataFrame(
        {
            "Open": [99.0, 99.0, 99.0, 99.0],
            "High": [101.0, 101.0, 101.0, 101.0],
            "Low": [98.0, 98.0, 98.0, 98.0],
            "Close": [99.0, 99.0, 99.0, 99.0],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=4,
            freq="D"
        )
    )

    class BuyStrategy:
        def generate_signals(self, data):

            signals = pd.Series(
                Signal.HOLD,
                index=data.index,
                dtype=object
            )

            signals.iloc[0] = Signal.BUY

            return signals

    sizer = VolatilityTargetSizer(
        target_volatility=0.15,
        max_fraction=1.0
    )

    engine = BacktestEngine(
        initial_capital=1000,
        transaction_cost=0.001,
        slippage=0.02,
        position_sizer=sizer,
        volatility_window=1
    )

    history, trades = engine.run(
        data,
        BuyStrategy()
    )

    # Most importantly, the backtest must not overspend.
    assert history["cash"].min() >= 0