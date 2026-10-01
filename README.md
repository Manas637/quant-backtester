# Quantitative Trading Strategy Backtester

A modular Python framework for researching and evaluating systematic trading strategies on historical market data.

The project implements a moving-average crossover strategy and evaluates it using transaction costs, slippage, position sizing, risk metrics, parameter selection, and walk-forward out-of-sample validation.

The main objective is to study how a simple systematic trading strategy behaves across different assets and market regimes while reducing the risk of look-ahead bias and parameter overfitting.

---

## Features

- Historical OHLCV market-data ingestion and validation
- Moving-average crossover strategy
- Event-driven backtesting engine
- Transaction-cost and slippage modeling
- Fixed-fraction position sizing
- Volatility-targeted position sizing
- Historical and rolling volatility estimation
- Performance and risk metrics
- Maximum drawdown analysis
- Trade-level statistics
- Training-only parameter sweeps
- Walk-forward out-of-sample evaluation
- Buy-and-hold benchmark comparison
- Parameter stability analysis
- Equity, drawdown, trade, and benchmark visualizations
- Unit tests for core components

---

## Project Structure

```text
quant-backtester/
│
├── data/
│   └── raw/
│
├── scripts/
│   ├── __init__.py
│   ├── download_data.py
│   ├── walk_forward.py
│   └── analyze_walk_forward.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── parameter_sweep.py
│   │   ├── data_split.py
│   │   └── model_selection.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── validator.py
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── moving_average.py
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── portfolio.py
│   │   ├── execution.py
│   │   └── position_sizing.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── performance.py
│   │   └── risk.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
│
├── tests/
│   ├── test_data.py
│   ├── test_strategy.py
│   ├── test_portfolio.py
│   ├── test_engine.py
│   ├── test_position_sizing.py
│   ├── test_performance.py
│   ├── test_execution.py
│   ├── test_visualization.py
│   ├── test_parameter_sweep.py
│   ├── test_data_split.py
│   └── test_model_selection.py
│
├── results/
│
├── run_backtest.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Market Data

The project works with daily OHLCV market data:

- Open
- High
- Low
- Close
- Volume

The current experiment evaluates the following assets:

- AAPL
- MSFT
- GOOGL
- AMZN
- SPY

Historical data is downloaded using `yfinance` and stored under:

```text
data/raw/
```

Raw market data is excluded from version control through `.gitignore`.

### 2. Data Validation

Before entering the backtesting engine, market data is validated for:

- Required OHLCV columns
- Duplicate timestamps
- Chronological ordering
- Missing values
- Positive OHLC prices
- Valid high/low relationships

This prevents malformed market data from entering the backtesting pipeline.

---

## Trading Strategy

The current strategy is a moving-average crossover strategy.

Two moving averages are calculated:

- Short-term moving average
- Long-term moving average

A trading signal is generated when the two moving averages cross.

### Buy Signal

A BUY signal occurs when:

```text
Short MA crosses above Long MA
```

### Sell Signal

A SELL signal occurs when:

```text
Short MA crosses below Long MA
```

Signals are generated using historical closing prices.

Orders are executed at the following trading day's market open.

This separates signal generation from execution and prevents the strategy from using the same closing price that generated the signal as the execution price.

---

## Backtesting Engine

The backtesting engine models a simple event-driven trading process.

For each trading day:

1. Process the signal generated using previously available information.
2. Execute the corresponding order at the current market open.
3. Apply slippage.
4. Apply transaction costs.
5. Update the portfolio.
6. Mark the portfolio to the current closing price.

The engine tracks:

- Cash
- Position quantity
- Portfolio value
- Trades
- Transaction costs
- Equity curve

---

## Transaction Costs and Slippage

The backtester supports transaction costs and execution slippage.

### Transaction Costs

A proportional transaction cost is applied to each trade.

The default research configuration uses:

```text
Transaction cost = 0.1%
```

### Slippage

Execution prices can be adjusted using:

```text
Buy price  = Market price × (1 + slippage)
Sell price = Market price × (1 - slippage)
```

The position-sizing logic accounts for the effective execution price so that the requested order remains affordable after slippage and transaction costs.

---

## Position Sizing

Two position-sizing approaches are implemented.

### Fixed-Fraction Sizing

A fixed percentage of available cash is allocated to a trade.

The default configuration uses:

```text
95% of available cash
```

### Volatility-Targeted Sizing

The volatility-targeted position sizer adjusts capital allocation based on estimated historical volatility.

The implementation uses the relationship:

```text
target allocation ≈ target volatility / estimated volatility
```

subject to a maximum allocation constraint.

The purpose is to investigate how volatility-aware sizing changes the risk/return characteristics of the same trading signals.

> The volatility-targeted implementation is a simplified inverse-volatility allocation heuristic. It is not a complete multi-asset portfolio volatility-targeting system.

---

## Parameter Selection

The moving-average strategy has two parameters:

```text
short_window
long_window
```

A parameter sweep evaluates combinations of these windows.

Combinations where:

```text
short_window >= long_window
```

are ignored.

The parameter sweep is performed only on training data.

The selected parameter set is the combination with the highest training Sharpe ratio.

The selected parameters are then evaluated on unseen test data.

This prevents the test period from influencing parameter selection.

---

## Walk-Forward Validation

To evaluate how the strategy behaves across different market regimes, the project uses walk-forward validation.

The experiment uses three out-of-sample periods:

| Fold | Training Data | Test Data |
|------|---------------|-----------|
| 1 | Before 2022 | 2022 |
| 2 | Before 2023 | 2023 |
| 3 | Before 2024 | 2024 |

For every fold:

1. Split the historical data into training and test periods.
2. Run the parameter sweep on training data.
3. Select the best parameter combination using training Sharpe ratio.
4. Run the selected strategy on the test period.
5. Evaluate out-of-sample performance.
6. Repeat for every asset.

The test data is never used for parameter selection.

---

## Performance Metrics

The project calculates several performance and risk metrics.

### Return Metrics

- Total return
- CAGR
- Buy-and-hold return

### Risk Metrics

- Annualized volatility
- Maximum drawdown
- Historical volatility
- Rolling historical volatility

### Risk-Adjusted Metrics

- Sharpe ratio

### Trade Statistics

- Number of trades
- Win rate
- Trade P&L
- Average winning trade
- Average losing trade
- Largest winning trade
- Largest losing trade
- Gross profit
- Gross loss
- Profit factor
- Average holding period

---

## Experimental Results

The walk-forward experiment was performed on:

- AAPL
- MSFT
- GOOGL
- AMZN
- SPY

using three out-of-sample folds.

### Fixed-Fraction Position Sizing

| Asset | Mean Return | Mean Fold Sharpe | Mean Volatility | Mean Max Drawdown | Profitable Folds |
|-------|------------:|-----------------:|----------------:|------------------:|-----------------:|
| AAPL  | 22.51% | 0.83 | 16.85% | -18.01% | 2/3 |
| AMZN  | 27.01% | 0.77 | 23.49% | -21.62% | 3/3 |
| GOOGL | 54.69% | 1.04 | 22.79% | -21.39% | 3/3 |
| MSFT  | 17.89% | 0.12 | 13.53% | -12.99% | 2/3 |
| SPY   | 18.55% | 1.00 | 7.35% | -6.31% | 2/3 |

> **Mean Fold Sharpe** is the arithmetic mean of the Sharpe ratios calculated independently for the three test folds. It is not a Sharpe ratio calculated from the combined return series.

### Volatility-Targeted Position Sizing

| Asset | Mean Return | Mean Fold Sharpe | Mean Volatility | Mean Max Drawdown | Profitable Folds |
|-------|------------:|-----------------:|----------------:|------------------:|-----------------:|
| AAPL  | 17.68% | 0.94 | 10.93% | -12.64% | 3/3 |
| AMZN  | 20.27% | 0.81 | 14.42% | -10.75% | 3/3 |
| GOOGL | 31.95% | 0.86 | 17.68% | -19.08% | 3/3 |
| MSFT  | 7.49% | 0.01 | 8.90% | -9.36% | 2/3 |
| SPY   | 19.05% | 1.08 | 6.85% | -5.52% | 2/3 |

---

## Position Sizing Analysis

The volatility-targeted sizing approach produced lower realized volatility and lower average maximum drawdown across all five assets in this experiment.

| Asset | Fixed Volatility | Volatility-Targeted Volatility | Volatility Reduction |
|-------|-----------------:|--------------------------------:|---------------------:|
| AAPL  | 16.85% | 10.93% | 5.92 pp |
| AMZN  | 23.49% | 14.42% | 9.07 pp |
| GOOGL | 22.79% | 17.68% | 5.12 pp |
| MSFT  | 13.53% | 8.90% | 4.63 pp |
| SPY   | 7.35% | 6.85% | 0.51 pp |

The change in risk-adjusted performance was asset-dependent.

For example:

- AAPL mean fold Sharpe increased from 0.83 to 0.94.
- AMZN increased from 0.77 to 0.81.
- GOOGL decreased from 1.04 to 0.86.
- MSFT decreased from 0.12 to approximately 0.01.
- SPY increased from 1.00 to 1.08.

Therefore, volatility-aware sizing changed the risk/return profile rather than uniformly improving every performance metric.

---

## Benchmark Comparison

The strategy was also compared against a buy-and-hold benchmark for each test fold.

The results show that the moving-average strategy did not consistently outperform buy-and-hold.

For example:

- GOOGL had positive mean excess return over buy-and-hold under fixed-fraction sizing.
- SPY also showed positive mean excess return under both sizing approaches.
- AAPL, AMZN, and MSFT had mixed results across individual folds.
- Some folds produced substantially different results depending on the market regime.

This highlights that the performance of a trend-following strategy can depend strongly on the underlying market regime.

---

## Parameter Stability

The selected moving-average parameters were not equally stable across all assets.

### AAPL

```text
10/20 → 5/20 → 5/20
```

2 unique parameter sets.

### AMZN

```text
5/20 → 10/20 → 5/20
```

2 unique parameter sets.

### GOOGL

```text
50/100 → 50/100 → 50/100
```

1 unique parameter set.

### MSFT

```text
50/100 → 5/100 → 30/200
```

3 unique parameter sets.

### SPY

```text
50/100 → 10/100 → 20/200
```

3 unique parameter sets.

The differences illustrate that parameter selection can be sensitive to the historical training period and underlying asset.

---

## Key Findings

### 1. Strategy performance is regime-dependent

The moving-average crossover strategy performed differently across the three test periods and across assets.

Historical performance should therefore not be interpreted as a universal expectation of future performance.

### 2. Volatility-targeted sizing reduced risk

Across all five assets, volatility-targeted sizing resulted in lower realized volatility and lower average maximum drawdown than fixed-fraction sizing.

### 3. Lower risk did not always mean higher returns

Volatility-targeted sizing generally reduced exposure during higher-volatility periods, which also reduced returns in several cases.

The effect on Sharpe ratio differed across assets.

### 4. Parameter stability varies by asset

GOOGL repeatedly selected the same parameter combination, while MSFT and SPY selected substantially different combinations across folds.

This demonstrates why evaluating parameter stability is useful when researching systematic strategies.

### 5. Buy-and-hold remains an important benchmark

A strategy should not be evaluated only on its absolute return.

Comparing against buy-and-hold helps determine how the systematic trading rule behaves relative to simply holding the asset.

---

## Avoiding Look-Ahead Bias

The project explicitly addresses several common sources of look-ahead bias.

### Signal Execution

Signals generated using historical closing prices are executed at the following trading day's open.

### Parameter Selection

Parameters are selected using training data only.

### Walk-Forward Testing

Each test period is unseen during parameter selection.

### Volatility Position Sizing

The volatility estimate used for position sizing is taken from information available before the trade is executed.

These constraints are important because allowing future information into a backtest can produce unrealistically strong results.

---

## Testing

The project includes unit tests covering:

- Market-data validation
- Data loading
- Strategy signal generation
- Portfolio accounting
- Order execution
- Position sizing
- Backtesting engine
- Performance metrics
- Risk metrics
- Parameter sweeps
- Data splitting
- Model selection
- Visualization functions

Run the complete test suite with:

```bash
python -m pytest -v
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Manas637/quant-backtester.git
cd quant-backtester
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Download Market Data

To download the research dataset:

```bash
python -m scripts.download_data
```

The script downloads daily data for:

- AAPL
- MSFT
- GOOGL
- AMZN
- SPY

and stores it under:

```text
data/raw/
```

### Run Walk-Forward Evaluation

```bash
python -m scripts.walk_forward
```

This performs:

- Train/test splitting
- Parameter sweeps
- Training-based parameter selection
- Fixed-fraction backtesting
- Volatility-targeted backtesting
- Out-of-sample evaluation
- Result generation

### Analyze Results

```bash
python -m scripts.analyze_walk_forward
```

This generates summary tables and visualizations under:

```text
results/
```

### Run a Single Backtest

A simple demonstration can be run with:

```bash
python run_backtest.py
```

The demonstration uses a fixed `20/50` moving-average configuration and is separate from the walk-forward parameter-selection experiment.

---

## Generated Results

The `results/` directory contains:

- `equity_curve.png`
- `price_with_trades.png`
- `walk_forward_return_by_asset.png`
- `walk_forward_sharpe_by_asset.png`
- `walk_forward_drawdown_by_asset.png`
- `walk_forward_fixed_vs_vol_target.png`
- `walk_forward_results.csv`
- `walk_forward_asset_summary.csv`
- `walk_forward_fold_summary.csv`
- `walk_forward_parameter_stability.csv`
- `walk_forward_comparison.csv`
- `walk_forward_vs_benchmark.csv`

These files provide visual and tabular representations of the experimental results.

---

## Limitations

This project is intended for research and educational purposes, not live trading.

Important limitations include:

- The strategy uses a relatively simple moving-average crossover.
- Only daily OHLCV data is used.
- The experiment covers a limited set of assets.
- Transaction costs and slippage are modeled using simplified assumptions.
- The volatility-targeting method is a simplified allocation heuristic.
- The backtester does not model market impact, bid-ask spreads, liquidity constraints, or partial fills.
- Corporate actions and data-quality issues depend on the underlying data provider.
- The parameter search space is intentionally limited.
- Historical backtest performance does not guarantee future performance.
- The experiment does not model taxes, financing costs, or exchange-specific trading constraints.

The results should therefore be interpreted as an empirical research exercise rather than evidence of a deployable trading system.

---

## Future Improvements

Potential extensions include:

- Additional systematic strategies
- Momentum and mean-reversion signals
- ATR-based position sizing
- Multi-asset portfolio construction
- Correlation-aware risk allocation
- Transaction-cost sensitivity analysis
- Parameter robustness testing
- Monte Carlo analysis
- Bootstrap confidence intervals
- Regime detection
- More realistic execution models
- Futures-specific contract handling
- Portfolio-level risk management

---

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- yfinance
- Pytest

---

## Disclaimer

This project is a quantitative research and backtesting framework created for educational and experimental purposes.

It does not constitute financial advice, an investment recommendation, or a guarantee of future trading performance.
