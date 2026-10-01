import pandas as pd

from src.backtesting.execution import ExecutionEngine
from src.backtesting.portfolio import Portfolio
from src.strategies.base import Signal, Strategy

from src.backtesting.position_sizing import (
    FixedFractionSizer,
    VolatilityTargetSizer
)

from src.metrics.risk import (
    rolling_historical_volatility
)


class BacktestEngine:

    def __init__(
        self,
        initial_capital,
        transaction_cost=0.001,
        position_fraction=0.95,
        close_positions=False,
        slippage=0.0,
        position_sizer=None,
        volatility_window=20
    ):

        if volatility_window <= 0:
            raise ValueError(
                "Volatility window must be positive"
            )

        self.volatility_window = volatility_window

        # Use custom position sizer if provided.
        # Otherwise use fixed-fraction sizing.
        if position_sizer is None:
            self.position_sizer = FixedFractionSizer(
                fraction=position_fraction
            )
        else:
            self.position_sizer = position_sizer

        self.portfolio = Portfolio(initial_capital)

        self.execution_engine = ExecutionEngine(
            transaction_cost=transaction_cost,
            slippage=slippage
        )

        self.close_positions = close_positions

    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        start_date=None
    ):

        # ------------------------------------------------------
        # Generate signals
        # ------------------------------------------------------

        # Generate signals on the complete dataset so indicators
        # can use historical warm-up data before start_date.
        signals = strategy.generate_signals(data)

        # ------------------------------------------------------
        # Calculate rolling volatility
        # ------------------------------------------------------

        rolling_volatility = rolling_historical_volatility(
            data["Close"],
            window=self.volatility_window
        )

        # ------------------------------------------------------
        # Determine evaluation start
        # ------------------------------------------------------

        if start_date is None:

            start_idx = 0

        else:

            start_date = pd.Timestamp(start_date)

            if start_date not in data.index:
                raise ValueError(
                    "start_date must be present in data index"
                )

            start_idx = data.index.get_loc(start_date)

        portfolio_history = []
        trades = []

        # ------------------------------------------------------
        # Main backtest loop
        # ------------------------------------------------------

        for i in range(start_idx, len(data)):

            current_date = data.index[i]

            # --------------------------------------------------
            # Execute yesterday's signal at today's Open
            # --------------------------------------------------

            if i > 0:

                previous_signal = signals.iloc[i - 1]

                market_open = data["Open"].iloc[i]

                # ==================================================
                # BUY
                # ==================================================

                if previous_signal == Signal.BUY:

                    # Only enter when currently flat.
                    if self.portfolio.quantity == 0:

                        # --------------------------------------------------
                        # IMPORTANT:
                        #
                        # The execution engine applies BUY slippage:
                        #
                        # execution_price =
                        #     market_open * (1 + slippage)
                        #
                        # Therefore the position sizer must use that
                        # expected execution price too.
                        # --------------------------------------------------

                        buy_execution_price = (
                            market_open *
                            (1 + self.execution_engine.slippage)
                        )

                        # ----------------------------------------------
                        # Volatility-targeted position sizing
                        # ----------------------------------------------

                        if isinstance(
                            self.position_sizer,
                            VolatilityTargetSizer
                        ):

                            # Use only information available before
                            # today's market open.
                            volatility = (
                                rolling_volatility.iloc[i - 1]
                            )

                            # Not enough historical observations.
                            if pd.isna(volatility):

                                quantity = 0

                            else:

                                quantity = (
                                    self.position_sizer.calculate_quantity(
                                        cash=self.portfolio.cash,
                                        price=buy_execution_price,
                                        volatility=volatility,
                                        transaction_cost_rate=(
                                            self.execution_engine
                                            .transaction_cost
                                        )
                                    )
                                )

                        # ----------------------------------------------
                        # Fixed-fraction position sizing
                        # ----------------------------------------------

                        else:

                            quantity = (
                                self.position_sizer.calculate_quantity(
                                    cash=self.portfolio.cash,
                                    price=buy_execution_price,
                                    transaction_cost_rate=(
                                        self.execution_engine
                                        .transaction_cost
                                    )
                                )
                            )

                        # ----------------------------------------------
                        # Execute trade
                        # ----------------------------------------------

                        if quantity > 0:

                            # Pass the ORIGINAL market price.
                            #
                            # ExecutionEngine applies slippage exactly
                            # once.
                            trade = self.execution_engine.execute(
                                date=current_date,
                                signal=Signal.BUY,
                                price=market_open,
                                quantity=quantity
                            )

                            self.portfolio.apply_trade(trade)

                            trades.append(trade)

                # ==================================================
                # SELL
                # ==================================================

                elif previous_signal == Signal.SELL:

                    # Sell the complete existing position.
                    if self.portfolio.quantity > 0:

                        quantity = self.portfolio.quantity

                        trade = self.execution_engine.execute(
                            date=current_date,
                            signal=Signal.SELL,
                            price=market_open,
                            quantity=quantity
                        )

                        self.portfolio.apply_trade(trade)

                        trades.append(trade)

            # ------------------------------------------------------
            # Mark portfolio to today's closing price
            # ------------------------------------------------------

            current_price = data["Close"].iloc[i]

            portfolio_history.append(
                {
                    "date": current_date,
                    "cash": self.portfolio.cash,
                    "quantity": self.portfolio.quantity,
                    "position_value": self.portfolio.position_value(
                        current_price
                    ),
                    "total_value": self.portfolio.total_value(
                        current_price
                    )
                }
            )

        # ----------------------------------------------------------
        # Force-close any open position at the end
        # ----------------------------------------------------------

        if (
            self.close_positions
            and self.portfolio.quantity > 0
        ):

            final_date = data.index[-1]
            final_price = data["Close"].iloc[-1]

            quantity = self.portfolio.quantity

            trade = self.execution_engine.execute(
                date=final_date,
                signal=Signal.SELL,
                price=final_price,
                quantity=quantity
            )

            self.portfolio.apply_trade(trade)

            trades.append(trade)

            # Update final snapshot after liquidation.
            portfolio_history[-1] = {
                "date": final_date,
                "cash": self.portfolio.cash,
                "quantity": self.portfolio.quantity,
                "position_value": self.portfolio.position_value(
                    final_price
                ),
                "total_value": self.portfolio.total_value(
                    final_price
                )
            }

        # ----------------------------------------------------------
        # Convert history to DataFrame
        # ----------------------------------------------------------

        history = pd.DataFrame(
            portfolio_history
        ).set_index("date")

        return history, trades