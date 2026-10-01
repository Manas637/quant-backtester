import pandas as pd
import pytest

from src.analysis.data_split import train_test_split


def create_test_data():
    return pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14],
            "High": [11, 12, 13, 14, 15],
            "Low": [9, 10, 11, 12, 13],
            "Close": [10, 11, 12, 13, 14],
            "Volume": [1000] * 5
        },
        index=pd.date_range(
            "2024-01-01",
            periods=5
        )
    )


def test_train_test_split():

    data = create_test_data()

    train, test = train_test_split(
        data,
        split_date="2024-01-04"
    )

    assert len(train) == 3
    assert len(test) == 2

    assert train.index[-1] == pd.Timestamp("2024-01-03")
    assert test.index[0] == pd.Timestamp("2024-01-04")


def test_empty_data():

    data = pd.DataFrame()

    with pytest.raises(ValueError):
        train_test_split(
            data,
            split_date="2024-01-01"
        )


def test_requires_datetime_index():

    data = pd.DataFrame(
        {
            "Close": [10, 11, 12]
        }
    )

    with pytest.raises(ValueError):
        train_test_split(
            data,
            split_date="2024-01-02"
        )


def test_empty_training_data():

    data = create_test_data()

    with pytest.raises(ValueError):
        train_test_split(
            data,
            split_date="2023-01-01"
        )


def test_empty_test_data():

    data = create_test_data()

    with pytest.raises(ValueError):
        train_test_split(
            data,
            split_date="2025-01-01"
        )