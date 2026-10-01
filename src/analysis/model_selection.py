import pandas as pd


def select_parameters(
    results: pd.DataFrame,
    metric: str = "sharpe_ratio"
) -> tuple[int, int]:

    if results.empty:
        raise ValueError("Parameter results cannot be empty")

    if metric not in results.columns:
        raise ValueError(
            f"Metric '{metric}' not found in results"
        )

    valid_results = results.dropna(subset=[metric])

    if valid_results.empty:
        raise ValueError(
            f"No valid results for metric '{metric}'"
        )

    best_row = valid_results.loc[
        valid_results[metric].idxmax()
    ]

    return (
        int(best_row["short_window"]),
        int(best_row["long_window"])
    )