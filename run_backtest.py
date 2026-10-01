from src.backtesting.engine import BacktestEngine
from src.data.loader import load_market_data
from src.metrics.performance import generate_performance_report
from src.strategies.moving_average import MovingAverageCrossover
from src.visualization.plots import (
    plot_drawdown,
    plot_equity_curve,
    plot_price_with_trades,
    plot_strategy_vs_benchmark,
)


def main():

    data = load_market_data(
        "data/raw/AAPL.csv"
    )

    strategy = MovingAverageCrossover(
        short_window=20,
        long_window=50
    )

    engine = BacktestEngine(
        initial_capital=100_000,
        transaction_cost=0.001,
        position_fraction=0.95,
        close_positions=True,
        slippage=0.001
    )

    history, trades = engine.run(
        data,
        strategy
    )

    report = generate_performance_report(
        history,
        trades,
        data
    )

    print("\nPERFORMANCE REPORT")
    print("-" * 40)

    for metric, value in report.items():
        print(f"{metric}: {value}")

    print("\nTRADES")
    print("-" * 40)

    for trade in trades:
        print(trade)

    plot_equity_curve(history)

    plot_drawdown(history)

    plot_price_with_trades(
        data,
        trades
    )

    plot_strategy_vs_benchmark(
        history,
        data
    )


if __name__ == "__main__":
    main()