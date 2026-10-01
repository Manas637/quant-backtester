import pandas as pd


def train_test_split(
    data: pd.DataFrame,
    split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if data.empty:
        raise ValueError("Market data cannot be empty")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex")

    split_date = pd.Timestamp(split_date)

    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]

    if train_data.empty:
        raise ValueError("Training data is empty")

    if test_data.empty:
        raise ValueError("Test data is empty")

    return train_data, test_data