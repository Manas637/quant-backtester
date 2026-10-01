import pandas as pd
import pytest

from src.strategies.base import Signal
from src.strategies.moving_average import MovingAverageCrossover

def test_invalid_short_window():
    with pytest.raises(ValueError):
        MovingAverageCrossover(
            short_window=0,
            long_window=5
        )


def test_invalid_long_window():
    with pytest.raises(ValueError):
        MovingAverageCrossover(
            short_window=3,
            long_window=0
        )


def test_short_window_must_be_smaller():
    with pytest.raises(ValueError):
        MovingAverageCrossover(
            short_window=10,
            long_window=5
        )

def test_missing_close_column():

    data = pd.DataFrame({
        "Open": [100, 101, 102]
    })

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3
    )

    with pytest.raises(ValueError):
        strategy.generate_signals(data)

def test_bullish_crossover():

    data = pd.DataFrame(
        {
            "Close": [
                10,
                10,
                10,
                20,
                30,
                40
            ]
        },
        index=pd.date_range(
            "2025-01-01",
            periods=6
        )
    )

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3
    )

    signals = strategy.generate_signals(data)

    assert Signal.BUY in signals.values

def test_bearish_crossover():

    data = pd.DataFrame(
        {
            "Close": [
                40,
                40,
                40,
                30,
                20,
                10
            ]
        },
        index=pd.date_range(
            "2025-01-01",
            periods=6
        )
    )

    strategy = MovingAverageCrossover(
        short_window=2,
        long_window=3
    )

    signals = strategy.generate_signals(data)

    assert Signal.SELL in signals.values