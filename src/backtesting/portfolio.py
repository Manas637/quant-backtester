from dataclasses import dataclass

from src.strategies.base import Signal
from src.backtesting.execution import Trade


@dataclass
class Portfolio:

    initial_capital: float

    cash: float = 0.0
    quantity: int = 0

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError(
                "Initial capital must be positive"
            )

        self.cash = self.initial_capital

    def apply_trade(self, trade: Trade) -> None:

        trade_value = (
            trade.price * trade.quantity
        )

        if trade.signal == Signal.BUY:

            total_cost = (
                trade_value +
                trade.transaction_cost
            )

            if total_cost > self.cash:
                raise ValueError(
                    "Insufficient cash"
                )

            self.cash -= total_cost
            self.quantity += trade.quantity

        elif trade.signal == Signal.SELL:

            if trade.quantity > self.quantity:
                raise ValueError(
                    "Cannot sell more shares than owned"
                )

            proceeds = (
                trade_value -
                trade.transaction_cost
            )

            self.cash += proceeds
            self.quantity -= trade.quantity

    def position_value(
        self,
        current_price: float
    ) -> float:

        if current_price < 0:
            raise ValueError(
                "Price cannot be negative"
            )

        return self.quantity * current_price

    def total_value(
        self,
        current_price: float
    ) -> float:

        return (
            self.cash +
            self.position_value(current_price)
        )