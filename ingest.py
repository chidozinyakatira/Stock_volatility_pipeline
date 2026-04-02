import yfinance as yf
import pandas as pd
import numpy as np

# The stocks we're comparing
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
START = "2018-01-01"
END = "2024-12-31"

def fetch_prices(tickers, start, end):
    print(f"Fetching data for: {tickers}")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = raw["Close"]
    print(f"Downloaded {len(prices)} rows")
    return prices

def calculate_returns(prices):
    # Log returns: log(P_t / P_t-1)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns

if __name__ == "__main__":
    prices = fetch_prices(TICKERS, START, END)
    returns = calculate_returns(prices)

    # Save both to CSV
    prices.to_csv("data/prices.csv")
    returns.to_csv("data/returns.csv")

    print("\nFirst 5 rows of log returns:")
    print(returns.head())
    print(f"\nShape: {returns.shape}")