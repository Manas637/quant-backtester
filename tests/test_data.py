import pandas as pd
import pytest

from src.data.validator import validate_market_data

from src.data.loader import load_market_data


def valid_data():
    return pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [98, 99, 100],
            "Close": [103, 104, 106],
            "Volume": [1000, 1200, 1500],
        },
        index=pd.to_datetime(
            ["2025-01-01", "2025-01-02", "2025-01-03"]
        ),
    )


def test_valid_market_data():
    df = valid_data()

    validate_market_data(df)


def test_missing_column():
    df = valid_data()
    df = df.drop(columns=["Volume"])

    with pytest.raises(ValueError):
        validate_market_data(df)


def test_duplicate_dates():
    df = valid_data()
    df.index = pd.to_datetime(
        ["2025-01-01", "2025-01-01", "2025-01-03"]
    )

    with pytest.raises(ValueError):
        validate_market_data(df)


def test_unsorted_dates():
    df = valid_data()
    df = df.iloc[::-1]

    with pytest.raises(ValueError):
        validate_market_data(df)


def test_missing_values():
    df = valid_data()
    df.loc[df.index[1], "Close"] = None

    with pytest.raises(ValueError):
        validate_market_data(df)


def test_invalid_price():
    df = valid_data()
    df.loc[df.index[1], "Close"] = -100

    with pytest.raises(ValueError):
        validate_market_data(df)

def test_load_market_data():
    df = load_market_data("data/raw/sample.csv")

    assert len(df) == 10
    assert list(df.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    assert df.index.is_monotonic_increasing
    assert df.index.is_unique

def test_loader_sorts_dates():
    df = load_market_data("data/raw/unsorted.csv")

    assert list(df.index) == list(
        pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03"
            ]
        )
    )

def test_invalid_high_price():
    df = valid_data()

    df.loc[df.index[1], "High"] = 90

    with pytest.raises(ValueError):
        validate_market_data(df)

def test_invalid_low_price():
    df = valid_data()

    df.loc[df.index[1], "Low"] = 200

    with pytest.raises(ValueError):
        validate_market_data(df)