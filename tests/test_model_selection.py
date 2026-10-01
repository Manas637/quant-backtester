import pandas as pd
import pytest

from src.analysis.model_selection import select_parameters


def test_select_parameters():

    results = pd.DataFrame(
        {
            "short_window": [5, 10, 20],
            "long_window": [20, 50, 100],
            "sharpe_ratio": [0.8, 1.2, 0.9]
        }
    )

    short, long = select_parameters(results)

    assert short == 10
    assert long == 50


def test_select_using_different_metric():

    results = pd.DataFrame(
        {
            "short_window": [5, 10, 20],
            "long_window": [20, 50, 100],
            "sharpe_ratio": [0.8, 1.2, 0.9],
            "cagr": [0.10, 0.15, 0.20]
        }
    )

    short, long = select_parameters(
        results,
        metric="cagr"
    )

    assert short == 20
    assert long == 100


def test_empty_results():

    results = pd.DataFrame()

    with pytest.raises(ValueError):
        select_parameters(results)


def test_invalid_metric():

    results = pd.DataFrame(
        {
            "short_window": [5],
            "long_window": [20],
            "sharpe_ratio": [1.0]
        }
    )

    with pytest.raises(ValueError):
        select_parameters(
            results,
            metric="invalid_metric"
        )