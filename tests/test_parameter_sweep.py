import pandas as pd

from src.analysis.parameter_sweep import run_parameter_sweep


def test_parameter_sweep():

    data = pd.DataFrame(
        {
            "Open": [
                10, 11, 12, 13, 14,
                15, 14, 13, 12, 11,
                12, 13, 14, 15, 16
            ],
            "High": [
                11, 12, 13, 14, 15,
                16, 15, 14, 13, 12,
                13, 14, 15, 16, 17
            ],
            "Low": [
                9, 10, 11, 12, 13,
                14, 13, 12, 11, 10,
                11, 12, 13, 14, 15
            ],
            "Close": [
                10, 11, 12, 13, 14,
                15, 14, 13, 12, 11,
                12, 13, 14, 15, 16
            ],
            "Volume": [1000] * 15
        },
        index=pd.date_range(
            "2025-01-01",
            periods=15
        )
    )

    results = run_parameter_sweep(
        data=data,
        short_windows=[2, 3],
        long_windows=[4, 5]
    )

    assert len(results) == 4

    assert "short_window" in results.columns
    assert "long_window" in results.columns
    assert "total_return" in results.columns
    assert "cagr" in results.columns
    assert "sharpe_ratio" in results.columns
    assert "max_drawdown" in results.columns