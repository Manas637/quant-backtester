from pathlib import Path

import pandas as pd

from src.data.loader import load_market_data
from src.analysis.parameter_sweep import run_parameter_sweep
from src.analysis.model_selection import select_parameters

from src.strategies.moving_average import MovingAverageCrossover

from src.backtesting.engine import BacktestEngine
from src.backtesting.position_sizing import VolatilityTargetSizer

from src.metrics.performance import generate_performance_report


DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "SPY"
]

INITIAL_CAPITAL = 100_000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.001

VOLATILITY_TARGET = 0.15
VOLATILITY_WINDOW = 20

SHORT_WINDOWS = [5, 10, 20, 30, 50]
LONG_WINDOWS = [20, 50, 100, 150, 200]


# ------------------------------------------------------------
# Walk-forward folds
# ------------------------------------------------------------

FOLDS = [
    {
        "train_end": "2022-01-01",
        "test_start": "2022-01-01",
        "test_end": "2023-01-01"
    },
    {
        "train_end": "2023-01-01",
        "test_start": "2023-01-01",
        "test_end": "2024-01-01"
    },
    {
        "train_end": "2024-01-01",
        "test_start": "2024-01-01",
        "test_end": "2025-01-01"
    }
]


def run_sizing_comparison(
    data,
    test_data,
    strategy
):
    """
    Run the same strategy using:

    1. Fixed 95% position sizing
    2. Volatility-targeted position sizing
    """

    # --------------------------------------------------------
    # Fixed-fraction sizing
    # --------------------------------------------------------

    fixed_engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        position_fraction=0.95,
        close_positions=True,
        slippage=SLIPPAGE
    )

    fixed_history, fixed_trades = fixed_engine.run(
        data,
        strategy,
        start_date=test_data.index[0]
    )

    fixed_report = generate_performance_report(
        fixed_history,
        fixed_trades,
        test_data
    )

    # --------------------------------------------------------
    # Volatility-targeted sizing
    # --------------------------------------------------------

    volatility_sizer = VolatilityTargetSizer(
        target_volatility=VOLATILITY_TARGET,
        max_fraction=1.0
    )

    volatility_engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        close_positions=True,
        slippage=SLIPPAGE,
        position_sizer=volatility_sizer,
        volatility_window=VOLATILITY_WINDOW
    )

    volatility_history, volatility_trades = (
        volatility_engine.run(
            data,
            strategy,
            start_date=test_data.index[0]
        )
    )

    volatility_report = generate_performance_report(
        volatility_history,
        volatility_trades,
        test_data
    )

    return fixed_report, volatility_report


def run_fold(
    ticker,
    data,
    fold_number,
    fold
):
    """
    Execute one walk-forward fold.
    """

    train_end = pd.Timestamp(
        fold["train_end"]
    )

    test_start = pd.Timestamp(
        fold["test_start"]
    )

    test_end = pd.Timestamp(
        fold["test_end"]
    )

    train_data = data[
        data.index < train_end
    ]

    test_data = data[
        (data.index >= test_start)
        & (data.index < test_end)
    ]

    if train_data.empty:
        raise ValueError(
            f"{ticker}: empty training period "
            f"for fold {fold_number}"
        )

    if test_data.empty:
        raise ValueError(
            f"{ticker}: empty test period "
            f"for fold {fold_number}"
        )

    print(
        f"\nFold {fold_number}: "
        f"train {train_data.index[0].date()} "
        f"-> {train_data.index[-1].date()}, "
        f"test {test_data.index[0].date()} "
        f"-> {test_data.index[-1].date()}"
    )

    # --------------------------------------------------------
    # Parameter selection using TRAIN only
    # --------------------------------------------------------

    sweep_results = run_parameter_sweep(
        data=train_data,
        short_windows=SHORT_WINDOWS,
        long_windows=LONG_WINDOWS,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        position_fraction=0.95,
        close_positions=True,
        slippage=SLIPPAGE
    )

    short_window, long_window = select_parameters(
        sweep_results,
        metric="sharpe_ratio"
    )

    print(
        f"Selected strategy: "
        f"{short_window}/{long_window}"
    )

    strategy = MovingAverageCrossover(
        short_window=short_window,
        long_window=long_window
    )

    # --------------------------------------------------------
    # OOS comparison
    # --------------------------------------------------------

    fixed_report, volatility_report = (
        run_sizing_comparison(
            data=data,
            test_data=test_data,
            strategy=strategy
        )
    )

    return {
        "ticker": ticker,
        "fold": fold_number,

        "train_start":
            train_data.index[0].date(),

        "train_end":
            train_data.index[-1].date(),

        "test_start":
            test_data.index[0].date(),

        "test_end":
            test_data.index[-1].date(),

        "short_window":
            short_window,

        "long_window":
            long_window,

        # Fixed sizing
        "fixed_total_return":
            fixed_report["total_return"],

        "fixed_cagr":
            fixed_report["cagr"],

        "fixed_volatility":
            fixed_report["annualized_volatility"],

        "fixed_sharpe":
            fixed_report["sharpe_ratio"],

        "fixed_max_drawdown":
            fixed_report["max_drawdown"],

        "fixed_trades":
            fixed_report["number_of_trades"],

        "fixed_win_rate":
            fixed_report["win_rate"],

        "fixed_profit_factor":
            fixed_report["profit_factor"],

        # Volatility targeting
        "vol_target_total_return":
            volatility_report["total_return"],

        "vol_target_cagr":
            volatility_report["cagr"],

        "vol_target_volatility":
            volatility_report["annualized_volatility"],

        "vol_target_sharpe":
            volatility_report["sharpe_ratio"],

        "vol_target_max_drawdown":
            volatility_report["max_drawdown"],

        "vol_target_trades":
            volatility_report["number_of_trades"],

        "vol_target_win_rate":
            volatility_report["win_rate"],

        "vol_target_profit_factor":
            volatility_report["profit_factor"],

        "buy_and_hold_return":
            fixed_report["buy_and_hold_return"]
    }


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for ticker in TICKERS:

        print("\n" + "=" * 80)
        print(f"WALK-FORWARD: {ticker}")
        print("=" * 80)

        data = load_market_data(
            DATA_DIR / f"{ticker}.csv"
        )

        for fold_number, fold in enumerate(
            FOLDS,
            start=1
        ):

            try:

                result = run_fold(
                    ticker=ticker,
                    data=data,
                    fold_number=fold_number,
                    fold=fold
                )

                all_results.append(result)

            except Exception as exc:

                print(
                    f"ERROR {ticker}, "
                    f"fold {fold_number}: {exc}"
                )

    if not all_results:
        raise RuntimeError(
            "No walk-forward results were generated."
        )

    results = pd.DataFrame(
        all_results
    )

    output_path = (
        RESULTS_DIR /
        "walk_forward_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS")
    print("=" * 80)

    print(
        results.to_string(index=False)
    )

    print(
        f"\nSaved results to {output_path}"
    )


if __name__ == "__main__":
    main()