import pandas as pd

from src.strategies.moving_average import MovingAverageCrossover
from src.backtesting.engine import BacktestEngine
from src.metrics.performance import generate_performance_report


def run_parameter_sweep(
    data,
    short_windows,
    long_windows,
    initial_capital=100000,
    transaction_cost=0.001,
    position_fraction=0.95,
    close_positions=True,
    slippage=0.0
):

    results = []

    for short_window in short_windows:

        for long_window in long_windows:

            if short_window >= long_window:
                continue

            strategy = MovingAverageCrossover(
                short_window=short_window,
                long_window=long_window
            )

            engine = BacktestEngine(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                position_fraction=position_fraction,
                close_positions=close_positions,
                slippage=slippage
            )

            history, trades = engine.run(
                data,
                strategy
            )

            try:

                report = generate_performance_report(
                    history,
                    trades,
                    data
                )

            except ValueError as exc:

                # Some parameter combinations can produce
                # zero-volatility equity curves, making Sharpe
                # undefined. Skip those combinations.
                if "Sharpe ratio is undefined" in str(exc):
                    continue

                raise

            results.append(
                {
                    "short_window": short_window,
                    "long_window": long_window,
                    **report
                }
            )

    return pd.DataFrame(results)