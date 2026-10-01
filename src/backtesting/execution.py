from dataclasses import dataclass
from datetime import datetime

from src.strategies.base import Signal


@dataclass
class Trade:
    date: datetime
    signal: Signal
    price: float
    quantity: int
    transaction_cost: float

class ExecutionEngine:

    def __init__(
        self,
        transaction_cost: float = 0.001,
        slippage: float = 0.0
    ):
        if transaction_cost < 0:
            raise ValueError(
                "Transaction cost cannot be negative"
            )

        if slippage < 0:
            raise ValueError(
                "Slippage cannot be negative"
            )

        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def execute(
        self,
        date,
        signal,
        price,
        quantity
    ) -> Trade:

        if price <= 0:
            raise ValueError("Price must be positive")

        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        if signal == Signal.BUY:
            execution_price = price * (1 + self.slippage)

        elif signal == Signal.SELL:
            execution_price = price * (1 - self.slippage)

        else:
            raise ValueError(
                "Execution signal must be BUY or SELL"
            )

        transaction_value = execution_price * quantity

        cost = (
            transaction_value *
            self.transaction_cost
        )

        return Trade(
            date=date,
            signal=signal,
            price=execution_price,
            quantity=quantity,
            transaction_cost=cost
        )