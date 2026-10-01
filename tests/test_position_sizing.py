import pytest

from src.backtesting.position_sizing import (
    FixedFractionSizer,
    VolatilityTargetSizer
)

def test_position_size():

    sizer = FixedFractionSizer(
        fraction=0.95
    )

    quantity = sizer.calculate_quantity(
        cash=1000,
        price=21,
        transaction_cost_rate=0
    )

    assert quantity == 45

def test_zero_cash():

    sizer = FixedFractionSizer(
        fraction=0.95
    )

    quantity = sizer.calculate_quantity(
        cash=0,
        price=21,
        transaction_cost_rate=0
    )

    assert quantity == 0

def test_invalid_fraction():

    with pytest.raises(ValueError):
        FixedFractionSizer(
            fraction=1.5
        )

def test_negative_fraction():

    with pytest.raises(ValueError):
        FixedFractionSizer(
            fraction=-0.1
        )

def test_invalid_price():

    sizer = FixedFractionSizer()

    with pytest.raises(ValueError):
        sizer.calculate_quantity(
            cash=1000,
            price=0,
            transaction_cost_rate=0
        )

def test_volatility_target_sizer():
    sizer = VolatilityTargetSizer(
        target_volatility=0.15,
        max_fraction=1.0
    )

    quantity = sizer.calculate_quantity(
        cash=100000,
        price=100,
        volatility=0.30
    )

    assert quantity == 500

def test_volatility_target_sizer_caps_allocation():

    sizer = VolatilityTargetSizer(
        target_volatility=0.15,
        max_fraction=1.0
    )

    quantity = sizer.calculate_quantity(
        cash=100000,
        price=100,
        volatility=0.05
    )

    assert quantity == 1000

def test_volatility_target_sizer_accounts_for_transaction_cost():

    sizer = VolatilityTargetSizer(
        target_volatility=0.15,
        max_fraction=1.0
    )

    quantity = sizer.calculate_quantity(
        cash=1000,
        price=100,
        volatility=0.15,
        transaction_cost_rate=0.01
    )

    assert quantity == 9

def test_volatility_target_sizer_invalid_target():

    with pytest.raises(ValueError):
        VolatilityTargetSizer(
            target_volatility=0
        )

def test_volatility_target_sizer_invalid_max_fraction():

    with pytest.raises(ValueError):
        VolatilityTargetSizer(
            target_volatility=0.15,
            max_fraction=1.5
        )

def test_volatility_target_sizer_invalid_volatility():

    sizer = VolatilityTargetSizer()

    with pytest.raises(ValueError):
        sizer.calculate_quantity(
            cash=100000,
            price=100,
            volatility=0
        )