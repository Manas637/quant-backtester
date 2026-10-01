from pathlib import Path

import yfinance as yf


TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "SPY"
]

START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

OUTPUT_DIR = Path("data/raw")


def download_ticker(ticker):

    print(f"\nDownloading {ticker}...")

    data = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for {ticker}"
        )

    # yfinance can return MultiIndex columns.
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data = data[
        ["Open", "High", "Low", "Close", "Volume"]
    ]

    output_path = OUTPUT_DIR / f"{ticker}.csv"

    data.to_csv(output_path)

    print(
        f"Saved {ticker}: "
        f"{len(data)} rows -> {output_path}"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for ticker in TICKERS:
        download_ticker(ticker)


if __name__ == "__main__":
    main()