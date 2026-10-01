import math

class FixedFractionSizer:

    def __init__(self, fraction: float = 0.95):

        if fraction <= 0 or fraction > 1:
            raise ValueError(
                "Fraction must be between 0 and 1"
            )

        self.fraction = fraction

    def calculate_quantity(
        self,
        cash: float,
        price: float,
        transaction_cost_rate: float
    ) -> int:

        if cash <= 0:
            return 0

        if price <= 0:
            raise ValueError(
                "Price must be positive"
            )

        if transaction_cost_rate < 0:
            raise ValueError(
                "Transaction cost rate cannot be negative"
            )

        capital_available = cash * self.fraction

        quantity = math.floor(
            capital_available / price
        )

        # Make sure transaction costs don't
        # cause us to exceed available cash.
        while quantity > 0:

            trade_value = quantity * price

            transaction_cost = (
                trade_value *
                transaction_cost_rate
            )

            total_cost = (
                trade_value +
                transaction_cost
            )

            if total_cost <= cash:
                break

            quantity -= 1

        return quantity

class VolatilityTargetSizer:
    def __init__(
        self,
        target_volatility: float = 0.15,
        max_fraction: float = 1.0
    ):
        if target_volatility <= 0:
            raise ValueError(
                "Target volatility must be positive"
            )

        if max_fraction <= 0 or max_fraction > 1:
            raise ValueError(
                "Max fraction must be in (0, 1]"
            )

        self.target_volatility = target_volatility
        self.max_fraction = max_fraction

    def calculate_quantity(
        self,
        cash: float,
        price: float,
        volatility: float,
        transaction_cost_rate: float = 0.0
    ) -> int:

        if cash < 0:
            raise ValueError(
                "Cash cannot be negative"
            )

        if price <= 0:
            raise ValueError(
                "Price must be positive"
            )

        if volatility <= 0:
            raise ValueError(
                "Volatility must be positive"
            )

        if transaction_cost_rate < 0:
            raise ValueError(
                "Transaction cost rate cannot be negative"
            )

        # Allocate less capital to more volatile assets.
        fraction = (
            self.target_volatility / volatility
        )

        # Never exceed the maximum allocation.
        fraction = min(
            fraction,
            self.max_fraction
        )

        capital_to_use = cash * fraction

        quantity = math.floor(
            capital_to_use / price
        )

        # Make sure transaction costs also fit
        # inside available cash.
        while quantity > 0:

            trade_value = quantity * price

            transaction_cost = (
                trade_value *
                transaction_cost_rate
            )

            total_cost = (
                trade_value +
                transaction_cost
            )

            if total_cost <= cash:
                break

            quantity -= 1

        return quantity