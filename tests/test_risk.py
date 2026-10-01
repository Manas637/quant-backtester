import pandas as pd
import pytest

from src.metrics.risk import (
    historical_volatility,
    rolling_historical_volatility
)

def test_historical_volatility():

    prices = pd.Series(
        [100, 101, 102, 101, 103, 104]
    )

    volatility = historical_volatility(
        prices,
        periods_per_year=252
    )

    assert volatility > 0

def test_historical_volatility_empty():

    prices = pd.Series(dtype=float)

    with pytest.raises(ValueError):
        historical_volatility(prices)

def test_historical_volatility_negative_price():

    prices = pd.Series(
        [100, 101, -102, 103]
    )

    with pytest.raises(ValueError):
        historical_volatility(prices)

def test_historical_volatility_invalid_periods():

    prices = pd.Series(
        [100, 101, 102]
    )

    with pytest.raises(ValueError):
        historical_volatility(
            prices,
            periods_per_year=0
        )

def test_historical_volatility_zero_volatility():

    prices = pd.Series(
        [100, 100, 100, 100, 100]
    )

    volatility = historical_volatility(
        prices
    )

    assert volatility == pytest.approx(0.0)

def test_rolling_historical_volatility():

    prices = pd.Series(
        [
            100,
            101,
            102,
            101,
            103,
            104,
            105,
            106,
            105,
            107
        ]
    )

    volatility = rolling_historical_volatility(
        prices,
        window=3,
        periods_per_year=252
    )

    # Not enough observations for the first
    # volatility estimate.
    assert volatility.iloc[0] != volatility.iloc[0]
    assert volatility.iloc[1] != volatility.iloc[1]
    assert volatility.iloc[2] != volatility.iloc[2]

    # From this point onward we should have
    # valid volatility estimates.
    assert volatility.iloc[3] > 0
    assert volatility.iloc[4] > 0

def test_rolling_historical_volatility_empty():

    prices = pd.Series(dtype=float)

    with pytest.raises(ValueError):
        rolling_historical_volatility(prices)

def test_rolling_historical_volatility_invalid_window():

    prices = pd.Series(
        [100, 101, 102, 103]
    )

    with pytest.raises(ValueError):
        rolling_historical_volatility(
            prices,
            window=0
        )

def test_rolling_historical_volatility_invalid_periods():

    prices = pd.Series(
        [100, 101, 102, 103]
    )

    with pytest.raises(ValueError):
        rolling_historical_volatility(
            prices,
            periods_per_year=0
        )

def test_rolling_historical_volatility_negative_price():

    prices = pd.Series(
        [100, 101, -102, 103]
    )

    with pytest.raises(ValueError):
        rolling_historical_volatility(prices)