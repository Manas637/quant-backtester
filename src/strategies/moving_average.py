import pandas as pd

from src.strategies.base import Signal, Strategy


class MovingAverageCrossover(Strategy):

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50
    ):
        if short_window <= 0:
            raise ValueError("Short window must be positive")

        if long_window <= 0:
            raise ValueError("Long window must be positive")

        if short_window >= long_window:
            raise ValueError(
                "Short window must be smaller than long window"
            )

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> pd.Series:

        if "Close" not in data.columns:
            raise ValueError(
                "Market data must contain a Close column"
            )

        close = data["Close"]

        short_ma = close.rolling(
            window=self.short_window
        ).mean()

        long_ma = close.rolling(
            window=self.long_window
        ).mean()

        signals = pd.Series(
            Signal.HOLD,
            index=data.index,
            dtype=object
        )

        cross_up = (
            (short_ma > long_ma) &
            (short_ma.shift(1) <= long_ma.shift(1))
        )

        cross_down = (
            (short_ma < long_ma) &
            (short_ma.shift(1) >= long_ma.shift(1))
        )

        signals[cross_up] = Signal.BUY
        signals[cross_down] = Signal.SELL

        return signals