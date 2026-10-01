import pandas as pd
import math

from src.strategies.base import Signal


def total_return(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty")

    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]

    if initial_value <= 0:
        raise ValueError("Initial portfolio value must be positive")

    return (final_value - initial_value) / initial_value

def cagr(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty")

    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]

    if initial_value <= 0:
        raise ValueError("Initial portfolio value must be positive")

    if final_value <= 0:
        raise ValueError("Final portfolio value must be positive")

    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]

    days = (end_date - start_date).days

    if days <= 0:
        raise ValueError("Equity curve must span more than one day")

    years = days / 365.25

    return (final_value / initial_value) ** (1 / years) - 1

def periodic_returns(equity_curve: pd.Series) -> pd.Series:
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("Portfolio values must be positive")

    return equity_curve.pct_change().dropna()

def annualized_volatility(
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive")

    returns = periodic_returns(equity_curve)

    if len(returns) < 2:
        raise ValueError(
            "At least two returns are required"
        )

    return returns.std(ddof=1) * math.sqrt(periods_per_year)

def sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive")

    if risk_free_rate < 0:
        raise ValueError("Risk-free rate cannot be negative")

    returns = periodic_returns(equity_curve)

    if len(returns) < 2:
        raise ValueError(
            "At least two returns are required"
        )

    periodic_rf = (
        (1 + risk_free_rate) ** (1 / periods_per_year)
    ) - 1

    excess_returns = returns - periodic_rf

    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        raise ValueError(
            "Sharpe ratio is undefined when volatility is zero"
        )

    return (
        excess_returns.mean() / volatility
    ) * math.sqrt(periods_per_year)

def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty")

    if (equity_curve <= 0).any():
        raise ValueError("Portfolio values must be positive")

    running_peak = equity_curve.cummax()

    drawdown = (
        equity_curve - running_peak
    ) / running_peak

    return drawdown.min()

def trade_returns(trades) -> list[float]:
    open_trade = None
    returns = []

    for trade in trades:

        if trade.signal == Signal.BUY:
            if open_trade is not None:
                raise ValueError(
                    "Multiple open positions are not supported"
                )

            open_trade = trade

        elif trade.signal == Signal.SELL:
            if open_trade is None:
                raise ValueError(
                    "SELL without an open position"
                )

            buy_value = (
                open_trade.price * open_trade.quantity
                + open_trade.transaction_cost
            )

            sell_value = (
                trade.price * trade.quantity
                - trade.transaction_cost
            )

            if trade.quantity != open_trade.quantity:
                raise ValueError(
                    "Partial position exits are not supported"
                )

            trade_return = (
                sell_value - buy_value
            ) / buy_value

            returns.append(trade_return)

            open_trade = None

    return returns

def win_rate(trades) -> float:
    returns = trade_returns(trades)

    if not returns:
        return 0.0

    winning_trades = sum(
        1 for trade_return in returns
        if trade_return > 0
    )

    return winning_trades / len(returns)

def trade_pnls(trades) -> list[float]:
    open_trade = None
    pnls = []

    for trade in trades:

        if trade.signal == Signal.BUY:
            if open_trade is not None:
                raise ValueError(
                    "Multiple open positions are not supported"
                )

            open_trade = trade

        elif trade.signal == Signal.SELL:
            if open_trade is None:
                raise ValueError(
                    "SELL without an open position"
                )

            if trade.quantity != open_trade.quantity:
                raise ValueError(
                    "Partial position exits are not supported"
                )

            buy_cost = (
                open_trade.price * open_trade.quantity
                + open_trade.transaction_cost
            )

            sell_proceeds = (
                trade.price * trade.quantity
                - trade.transaction_cost
            )

            pnls.append(
                sell_proceeds - buy_cost
            )

            open_trade = None

    return pnls

def number_of_trades(trades) -> int:
    pnls = trade_pnls(trades)
    return len(pnls)


def average_winning_trade(trades) -> float:
    pnls = trade_pnls(trades)

    winning_pnls = [
        pnl for pnl in pnls
        if pnl > 0
    ]

    if not winning_pnls:
        return 0.0

    return sum(winning_pnls) / len(winning_pnls)


def average_losing_trade(trades) -> float:
    pnls = trade_pnls(trades)

    losing_pnls = [
        pnl for pnl in pnls
        if pnl < 0
    ]

    if not losing_pnls:
        return 0.0

    return sum(losing_pnls) / len(losing_pnls)


def largest_winning_trade(trades) -> float:
    pnls = trade_pnls(trades)

    winning_pnls = [
        pnl for pnl in pnls
        if pnl > 0
    ]

    if not winning_pnls:
        return 0.0

    return max(winning_pnls)


def largest_losing_trade(trades) -> float:
    pnls = trade_pnls(trades)

    losing_pnls = [
        pnl for pnl in pnls
        if pnl < 0
    ]

    if not losing_pnls:
        return 0.0

    return min(losing_pnls)


def gross_profit(trades) -> float:
    pnls = trade_pnls(trades)

    return sum(
        pnl for pnl in pnls
        if pnl > 0
    )


def gross_loss(trades) -> float:
    pnls = trade_pnls(trades)

    return abs(
        sum(
            pnl for pnl in pnls
            if pnl < 0
        )
    )

def average_holding_period(trades) -> float:
    open_trade = None
    holding_periods = []

    for trade in trades:

        if trade.signal == Signal.BUY:

            if open_trade is not None:
                raise ValueError(
                    "Multiple open positions are not supported"
                )

            open_trade = trade

        elif trade.signal == Signal.SELL:

            if open_trade is None:
                raise ValueError(
                    "SELL without an open position"
                )

            if trade.quantity != open_trade.quantity:
                raise ValueError(
                    "Partial position exits are not supported"
                )

            holding_days = (
                trade.date - open_trade.date
            ).days

            holding_periods.append(holding_days)

            open_trade = None

    if not holding_periods:
        return 0.0

    return sum(holding_periods) / len(holding_periods)

def profit_factor(trades) -> float:
    pnls = trade_pnls(trades)

    if not pnls:
        return 0.0

    gross_profit = sum(
        pnl for pnl in pnls
        if pnl > 0
    )

    gross_loss = abs(
        sum(
            pnl for pnl in pnls
            if pnl < 0
        )
    )

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")

        return 0.0

    return gross_profit / gross_loss

def buy_and_hold_return(data: pd.DataFrame) -> float:
    if "Close" not in data.columns:
        raise ValueError(
            "Data must contain a Close column"
        )

    if data.empty:
        raise ValueError("Data cannot be empty")

    initial_price = data["Close"].iloc[0]
    final_price = data["Close"].iloc[-1]

    if initial_price <= 0:
        raise ValueError("Initial price must be positive")

    if final_price <= 0:
        raise ValueError("Final price must be positive")

    return (
        final_price - initial_price
    ) / initial_price

def generate_performance_report(
    history: pd.DataFrame,
    trades,
    data: pd.DataFrame
) -> dict:

    if "total_value" not in history.columns:
        raise ValueError(
            "History must contain a total_value column"
        )

    equity_curve = history["total_value"]

    annualized_vol = annualized_volatility(
        equity_curve
    )

    # Sharpe ratio is undefined when portfolio
    # return volatility is zero.
    #
    # We preserve NaN rather than converting it to 0,
    # because 0 and undefined are different cases.
    if annualized_vol == 0:
        sharpe = float("nan")
    else:
        sharpe = sharpe_ratio(
            equity_curve
        )

    return {
        "total_return": total_return(
            equity_curve
        ),

        "cagr": cagr(
            equity_curve
        ),

        "annualized_volatility": annualized_vol,

        "sharpe_ratio": sharpe,

        "max_drawdown": max_drawdown(
            equity_curve
        ),

        "number_of_trades": number_of_trades(
            trades
        ),

        "win_rate": win_rate(
            trades
        ),

        "profit_factor": profit_factor(
            trades
        ),

        "gross_profit": gross_profit(
            trades
        ),

        "gross_loss": gross_loss(
            trades
        ),

        "average_winning_trade":
            average_winning_trade(trades),

        "average_losing_trade":
            average_losing_trade(trades),

        "largest_winning_trade":
            largest_winning_trade(trades),

        "largest_losing_trade":
            largest_losing_trade(trades),

        "average_holding_period_days":
            average_holding_period(trades),

        "buy_and_hold_return":
            buy_and_hold_return(data)
    }