import math
import pandas as pd


def historical_volatility(
    prices: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized historical volatility
    using close-to-close returns.
    """

    if prices.empty:
        raise ValueError("Price series cannot be empty")

    if periods_per_year <= 0:
        raise ValueError(
            "Periods per year must be positive"
        )

    if (prices <= 0).any():
        raise ValueError(
            "Prices must be positive"
        )

    returns = prices.pct_change().dropna()

    if len(returns) < 2:
        raise ValueError(
            "At least three prices are required"
        )

    daily_volatility = returns.std(ddof=1)

    return daily_volatility * math.sqrt(
        periods_per_year
    )

def rolling_historical_volatility(
    prices: pd.Series,
    window: int = 20,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate rolling annualized historical volatility
    using close-to-close returns.

    Each volatility estimate uses only prices available
    up to that date.
    """

    if prices.empty:
        raise ValueError("Price series cannot be empty")

    if window <= 0:
        raise ValueError(
            "Window must be positive"
        )

    if periods_per_year <= 0:
        raise ValueError(
            "Periods per year must be positive"
        )

    if (prices <= 0).any():
        raise ValueError(
            "Prices must be positive"
        )

    returns = prices.pct_change()

    volatility = (
        returns
        .rolling(window=window)
        .std(ddof=1)
        * math.sqrt(periods_per_year)
    )

    return volatility