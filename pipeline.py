import os
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from spectral import run_all_spectral
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
START = "2018-01-01"
END = "2024-12-31"
DATA_DIR = "data"

# --- STEP 1: INGEST ---
def fetch_and_save():
    print("=" * 40)
    print("STEP 1: Fetching price data...")
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)
    prices = raw["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()

    prices.to_csv(f"{DATA_DIR}/prices.csv")
    log_returns.to_csv(f"{DATA_DIR}/returns.csv")

    print(f"  Prices: {prices.shape}")
    print(f"  Returns: {log_returns.shape}")
    return log_returns

# --- STEP 2: MODEL ---
def fit_garch(returns_series, ticker):
    r = returns_series.dropna() * 100
    model = arch_model(r, vol="Garch", p=1, q=1, dist="normal")
    result = model.fit(disp="off")
    cond_vol = result.conditional_volatility * np.sqrt(252) / 100
    params = result.params
    return {
        "ticker": ticker,
        "omega": params["omega"],
        "alpha": params["alpha[1]"],
        "beta": params["beta[1]"],
        "persistence": params["alpha[1]"] + params["beta[1]"],
        "var_95": float(np.percentile(r / 100, 5)),
        "cond_vol": cond_vol
    }

def model_and_save(returns):
    print("=" * 40)
    print("STEP 2: Fitting GARCH models...")
    all_vol = {}
    summary_rows = []

    for ticker in returns.columns:
        print(f"  Fitting {ticker}...")
        result = fit_garch(returns[ticker], ticker)
        all_vol[ticker] = result["cond_vol"]
        summary_rows.append({k: v for k, v in result.items() if k != "cond_vol"})

    vol_df = pd.DataFrame(all_vol)
    vol_df.index = returns.index[len(returns.index) - len(vol_df):]
    vol_df.to_csv(f"{DATA_DIR}/volatility.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{DATA_DIR}/model_summary.csv", index=False)

    print(f"  Volatility series saved: {vol_df.shape}")
    print(f"  Model summary saved: {summary_df.shape}")

# --- STEP 3: REPORT ---
def report():
    print("=" * 40)
    print("STEP 3: Pipeline summary")
    summary = pd.read_csv(f"{DATA_DIR}/model_summary.csv")
    most_volatile = summary.loc[summary["var_95"].idxmin(), "ticker"]
    most_persistent = summary.loc[summary["persistence"].idxmax(), "ticker"]
    print(f"  Most volatile (VaR):     {most_volatile}")
    print(f"  Most persistent (a+b):   {most_persistent}")
    print("  Pipeline complete.")
    print("=" * 40)

# --- MAIN ---
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    returns = fetch_and_save()
    model_and_save(returns)
    report()
    run_all_spectral()
    print("All steps complete.")

