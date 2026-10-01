from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1


class Strategy(ABC):

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate trading signals from market data.

        Returns:
            pd.Series containing Signal values.
        """
        pass