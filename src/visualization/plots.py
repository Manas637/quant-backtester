import matplotlib.pyplot as plt
import pandas as pd

from src.strategies.base import Signal


def plot_equity_curve(
    history: pd.DataFrame,
    show: bool = True
):
    if "total_value" not in history.columns:
        raise ValueError(
            "History must contain a total_value column"
        )

    if history.empty:
        raise ValueError("History cannot be empty")

    fig, ax = plt.subplots()

    ax.plot(
        history.index,
        history["total_value"],
        label="Portfolio Value"
    )

    ax.set_title("Portfolio Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)

    fig.autofmt_xdate()

    if show:
        plt.show()

    return fig, ax

def plot_drawdown(
    history: pd.DataFrame,
    show: bool = True
):
    if "total_value" not in history.columns:
        raise ValueError(
            "History must contain a total_value column"
        )

    if history.empty:
        raise ValueError("History cannot be empty")

    running_peak = history["total_value"].cummax()

    drawdown = (
        history["total_value"] - running_peak
    ) / running_peak

    fig, ax = plt.subplots()

    ax.plot(
        history.index,
        drawdown,
        label="Drawdown"
    )

    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True)

    fig.autofmt_xdate()

    if show:
        plt.show()

    return fig, ax


def plot_price_with_trades(
    data: pd.DataFrame,
    trades,
    show: bool = True
):
    if "Close" not in data.columns:
        raise ValueError(
            "Data must contain a Close column"
        )

    if data.empty:
        raise ValueError("Data cannot be empty")

    fig, ax = plt.subplots()

    ax.plot(
        data.index,
        data["Close"],
        label="Close Price"
    )

    buy_label_used = False
    sell_label_used = False

    for trade in trades:

        if trade.date not in data.index:
            continue

        if trade.signal == Signal.BUY:
            marker = "^"

            label = "BUY" if not buy_label_used else "_nolegend_"
            buy_label_used = True

        elif trade.signal == Signal.SELL:
            marker = "v"

            label = "SELL" if not sell_label_used else "_nolegend_"
            sell_label_used = True

        else:
            continue

        ax.scatter(
            trade.date,
            trade.price,
            marker=marker,
            label=label
        )

    ax.set_title("Price and Trade Executions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    fig.autofmt_xdate()

    if show:
        plt.show()

    return fig, ax

def plot_strategy_vs_benchmark(
    history: pd.DataFrame,
    data: pd.DataFrame,
    show: bool = True
):
    if "total_value" not in history.columns:
        raise ValueError(
            "History must contain a total_value column"
        )

    if "Close" not in data.columns:
        raise ValueError(
            "Data must contain a Close column"
        )

    if history.empty:
        raise ValueError("History cannot be empty")

    if data.empty:
        raise ValueError("Data cannot be empty")

    # Use only the price data corresponding to the
    # evaluation period.
    benchmark_data = data.loc[
        data.index.intersection(history.index)
    ]

    if benchmark_data.empty:
        raise ValueError(
            "No overlapping dates between history and data"
        )

    initial_capital = history["total_value"].iloc[0]

    initial_price = benchmark_data["Close"].iloc[0]

    benchmark = (
        benchmark_data["Close"] / initial_price
    ) * initial_capital

    fig, ax = plt.subplots()

    ax.plot(
        history.index,
        history["total_value"],
        label="Strategy"
    )

    ax.plot(
        benchmark_data.index,
        benchmark,
        label="Buy & Hold"
    )

    ax.set_title(
        "Strategy vs Buy-and-Hold"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")

    ax.legend()
    ax.grid(True)

    fig.autofmt_xdate()

    if show:
        plt.show()

    return fig, ax