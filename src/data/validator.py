import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


def validate_market_data(df: pd.DataFrame) -> None:

    # 1. Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 2. Check duplicate dates
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found")

    # 3. Check chronological order
    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted")

    # 4. Check missing values
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing values found")

    # 5. Check invalid prices
    price_columns = ["Open", "High", "Low", "Close"]

    if (df[price_columns] <= 0).any().any():
        raise ValueError("Invalid price found")

    if (df["High"] < df["Open"]).any():
        raise ValueError("High price cannot be below Open price")

    if (df["High"] < df["Close"]).any():
        raise ValueError("High price cannot be below Close price")

    if (df["Low"] > df["Open"]).any():
        raise ValueError("Low price cannot be above Open price")

    if (df["Low"] > df["Close"]).any():
        raise ValueError("Low price cannot be above Close price")