import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "results/walk_forward_results.csv"

ASSET_SUMMARY_FILE = (
    "results/walk_forward_asset_summary.csv"
)

FOLD_SUMMARY_FILE = (
    "results/walk_forward_fold_summary.csv"
)

PARAMETER_FILE = (
    "results/walk_forward_parameter_stability.csv"
)

COMPARISON_FILE = (
    "results/walk_forward_comparison.csv"
)


def load_results():

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "Walk-forward results file is empty"
        )

    required_columns = [
        "ticker",
        "fold",
        "short_window",
        "long_window",

        "fixed_total_return",
        "fixed_volatility",
        "fixed_sharpe",
        "fixed_max_drawdown",
        "fixed_trades",

        "vol_target_total_return",
        "vol_target_volatility",
        "vol_target_sharpe",
        "vol_target_max_drawdown",
        "vol_target_trades",

        "buy_and_hold_return"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


def analyze_by_asset(df):

    rows = []

    for ticker, group in df.groupby("ticker"):

        fixed_returns = group[
            "fixed_total_return"
        ]

        vt_returns = group[
            "vol_target_total_return"
        ]

        fixed_sharpe = group[
            "fixed_sharpe"
        ].dropna()

        vt_sharpe = group[
            "vol_target_sharpe"
        ].dropna()

        rows.append(
            {
                "ticker": ticker,

                "folds": len(group),

                "fixed_mean_return":
                    fixed_returns.mean(),

                "fixed_median_return":
                    fixed_returns.median(),

                "fixed_mean_sharpe":
                    fixed_sharpe.mean(),

                "fixed_median_sharpe":
                    fixed_sharpe.median(),

                "fixed_mean_volatility":
                    group[
                        "fixed_volatility"
                    ].mean(),

                "fixed_mean_max_drawdown":
                    group[
                        "fixed_max_drawdown"
                    ].mean(),

                "fixed_profitable_folds":
                    (fixed_returns > 0).sum(),

                "fixed_profitable_fold_rate":
                    (fixed_returns > 0).mean(),

                "fixed_total_trades":
                    group[
                        "fixed_trades"
                    ].sum(),

                "vol_target_mean_return":
                    vt_returns.mean(),

                "vol_target_median_return":
                    vt_returns.median(),

                "vol_target_mean_sharpe":
                    vt_sharpe.mean(),

                "vol_target_median_sharpe":
                    vt_sharpe.median(),

                "vol_target_mean_volatility":
                    group[
                        "vol_target_volatility"
                    ].mean(),

                "vol_target_mean_max_drawdown":
                    group[
                        "vol_target_max_drawdown"
                    ].mean(),

                "vol_target_profitable_folds":
                    (vt_returns > 0).sum(),

                "vol_target_profitable_fold_rate":
                    (vt_returns > 0).mean(),

                "vol_target_total_trades":
                    group[
                        "vol_target_trades"
                    ].sum(),

                "buy_and_hold_mean_return":
                    group[
                        "buy_and_hold_return"
                    ].mean(),

                "buy_and_hold_median_return":
                    group[
                        "buy_and_hold_return"
                    ].median()
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        ASSET_SUMMARY_FILE,
        index=False
    )

    return result


def analyze_by_fold(df):

    rows = []

    for fold, group in df.groupby("fold"):

        fixed_returns = group[
            "fixed_total_return"
        ]

        vt_returns = group[
            "vol_target_total_return"
        ]

        rows.append(
            {
                "fold": fold,

                "fixed_mean_return":
                    fixed_returns.mean(),

                "fixed_median_return":
                    fixed_returns.median(),

                "fixed_mean_sharpe":
                    group[
                        "fixed_sharpe"
                    ].mean(),

                "fixed_mean_volatility":
                    group[
                        "fixed_volatility"
                    ].mean(),

                "fixed_mean_max_drawdown":
                    group[
                        "fixed_max_drawdown"
                    ].mean(),

                "fixed_profitable_assets":
                    (fixed_returns > 0).sum(),

                "fixed_profitable_asset_rate":
                    (fixed_returns > 0).mean(),

                "vol_target_mean_return":
                    vt_returns.mean(),

                "vol_target_median_return":
                    vt_returns.median(),

                "vol_target_mean_sharpe":
                    group[
                        "vol_target_sharpe"
                    ].mean(),

                "vol_target_mean_volatility":
                    group[
                        "vol_target_volatility"
                    ].mean(),

                "vol_target_mean_max_drawdown":
                    group[
                        "vol_target_max_drawdown"
                    ].mean(),

                "vol_target_profitable_assets":
                    (vt_returns > 0).sum(),

                "vol_target_profitable_asset_rate":
                    (vt_returns > 0).mean(),

                "buy_and_hold_mean_return":
                    group[
                        "buy_and_hold_return"
                    ].mean()
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        FOLD_SUMMARY_FILE,
        index=False
    )

    return result


def analyze_parameter_stability(df):

    rows = []

    for ticker, group in df.groupby("ticker"):

        unique_parameters = (
            group[
                [
                    "short_window",
                    "long_window"
                ]
            ]
            .drop_duplicates()
        )

        parameter_strings = (
            group[
                [
                    "short_window",
                    "long_window"
                ]
            ]
            .astype(str)
            .agg(
                "/".join,
                axis=1
            )
            .tolist()
        )

        rows.append(
            {
                "ticker": ticker,

                "folds": len(group),

                "unique_parameter_sets":
                    len(unique_parameters),

                "parameter_stability_rate":
                    1 -
                    (
                        len(unique_parameters) - 1
                    ) / len(group),

                "selected_parameters":
                    " -> ".join(
                        parameter_strings
                    )
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        PARAMETER_FILE,
        index=False
    )

    return result

def analyze_vs_benchmark(df):

    rows = []

    for ticker, group in df.groupby("ticker"):

        fixed_excess = (
            group["fixed_total_return"]
            - group["buy_and_hold_return"]
        )

        vt_excess = (
            group["vol_target_total_return"]
            - group["buy_and_hold_return"]
        )

        rows.append(
            {
                "ticker": ticker,

                "fixed_mean_excess_return":
                    fixed_excess.mean(),

                "fixed_median_excess_return":
                    fixed_excess.median(),

                "fixed_folds_beating_benchmark":
                    (fixed_excess > 0).sum(),

                "fixed_benchmark_win_rate":
                    (fixed_excess > 0).mean(),

                "vol_target_mean_excess_return":
                    vt_excess.mean(),

                "vol_target_median_excess_return":
                    vt_excess.median(),

                "vol_target_folds_beating_benchmark":
                    (vt_excess > 0).sum(),

                "vol_target_benchmark_win_rate":
                    (vt_excess > 0).mean()
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        "results/walk_forward_vs_benchmark.csv",
        index=False
    )

    return result

def compare_methods(df):

    rows = []

    for ticker, group in df.groupby("ticker"):

        fixed_return = (
            group[
                "fixed_total_return"
            ].mean()
        )

        vt_return = (
            group[
                "vol_target_total_return"
            ].mean()
        )

        fixed_vol = (
            group[
                "fixed_volatility"
            ].mean()
        )

        vt_vol = (
            group[
                "vol_target_volatility"
            ].mean()
        )

        fixed_sharpe = (
            group[
                "fixed_sharpe"
            ].mean()
        )

        vt_sharpe = (
            group[
                "vol_target_sharpe"
            ].mean()
        )

        fixed_dd = (
            group[
                "fixed_max_drawdown"
            ].mean()
        )

        vt_dd = (
            group[
                "vol_target_max_drawdown"
            ].mean()
        )

        rows.append(
            {
                "ticker": ticker,

                "fixed_mean_return":
                    fixed_return,

                "vol_target_mean_return":
                    vt_return,

                "return_difference":
                    vt_return - fixed_return,

                "fixed_mean_volatility":
                    fixed_vol,

                "vol_target_mean_volatility":
                    vt_vol,

                "volatility_reduction":
                    fixed_vol - vt_vol,

                "fixed_mean_sharpe":
                    fixed_sharpe,

                "vol_target_mean_sharpe":
                    vt_sharpe,

                "sharpe_difference":
                    vt_sharpe - fixed_sharpe,

                "fixed_mean_drawdown":
                    fixed_dd,

                "vol_target_mean_drawdown":
                    vt_dd,

                "drawdown_improvement":
                    vt_dd - fixed_dd
            }
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        COMPARISON_FILE,
        index=False
    )

    return result


def plot_return_by_asset(df):

    pivot = df.pivot(
        index="ticker",
        columns="fold",
        values="fixed_total_return"
    )

    pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Fixed-Fraction Test Returns by Asset and Fold"
    )

    plt.xlabel("Asset")
    plt.ylabel("Test Return")

    plt.tight_layout()

    plt.savefig(
        "results/walk_forward_return_by_asset.png"
    )

    plt.close()


def plot_sharpe_by_asset(df):

    pivot = df.pivot(
        index="ticker",
        columns="fold",
        values="fixed_sharpe"
    )

    pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Fixed-Fraction Test Sharpe by Asset and Fold"
    )

    plt.xlabel("Asset")
    plt.ylabel("Sharpe Ratio")

    plt.tight_layout()

    plt.savefig(
        "results/walk_forward_sharpe_by_asset.png"
    )

    plt.close()


def plot_drawdown_by_asset(df):

    pivot = df.pivot(
        index="ticker",
        columns="fold",
        values="fixed_max_drawdown"
    )

    pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Fixed-Fraction Maximum Drawdown"
    )

    plt.xlabel("Asset")
    plt.ylabel("Maximum Drawdown")

    plt.tight_layout()

    plt.savefig(
        "results/walk_forward_drawdown_by_asset.png"
    )

    plt.close()


def plot_position_sizing_comparison(df):

    grouped = df.groupby("ticker")[
        [
            "fixed_total_return",
            "vol_target_total_return"
        ]
    ].mean()

    grouped.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Fixed Fraction vs Volatility Targeting"
    )

    plt.xlabel("Asset")
    plt.ylabel("Mean Test Return")

    plt.tight_layout()

    plt.savefig(
        "results/walk_forward_fixed_vs_vol_target.png"
    )

    plt.close()


def main():

    df = load_results()

    asset_summary = analyze_by_asset(df)

    fold_summary = analyze_by_fold(df)

    parameter_summary = (
        analyze_parameter_stability(df)
    )

    comparison = compare_methods(df)

    vs_benchmark = analyze_vs_benchmark(df)

    plot_return_by_asset(df)

    plot_sharpe_by_asset(df)

    plot_drawdown_by_asset(df)

    plot_position_sizing_comparison(df)

    print()
    print("=" * 80)
    print("WALK-FORWARD ASSET SUMMARY")
    print("=" * 80)

    print(
        asset_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("WALK-FORWARD FOLD SUMMARY")
    print("=" * 80)

    print(
        fold_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("PARAMETER STABILITY")
    print("=" * 80)

    print(
        parameter_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("POSITION SIZING COMPARISON")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("Analysis completed.")
    print()
    print(
        f"Saved asset summary to {ASSET_SUMMARY_FILE}"
    )
    print(
        f"Saved fold summary to {FOLD_SUMMARY_FILE}"
    )
    print(
        f"Saved parameter summary to {PARAMETER_FILE}"
    )
    print(
        f"Saved comparison to {COMPARISON_FILE}"
    )

    print()
    print("=" * 80)
    print("STRATEGY VS BUY-AND-HOLD")
    print("=" * 80)

    print(
        vs_benchmark.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()