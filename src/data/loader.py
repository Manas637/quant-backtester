import pandas as pd

from src.data.validator import validate_market_data


def load_market_data(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV market data from a CSV file.

    Expected columns:
        Date, Open, High, Low, Close, Volume
    """

    df = pd.read_csv(filepath)

    if "Date" not in df.columns:
        raise ValueError("CSV must contain a Date column")

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Use Date as the DataFrame index
    df = df.set_index("Date")

    # Sort chronologically
    df = df.sort_index()

    # Validate the cleaned data
    validate_market_data(df)

    return df