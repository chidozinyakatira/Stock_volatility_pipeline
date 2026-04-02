import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

def fit_garch(returns_series, ticker):
    """Fit GARCH(1,1) to a single stock's return series."""
    # Scale returns by 100 -- arch library works better this way
    r = returns_series.dropna() * 100

    model = arch_model(r, vol="Garch", p=1, q=1, dist="normal")
    result = model.fit(disp="off")

    # Extract conditional volatility (annualised)
    cond_vol = result.conditional_volatility * np.sqrt(252) / 100

    # Extract parameters
    params = result.params

    # Value at Risk (95%) -- daily
    var_95 = float(np.percentile(r / 100, 5))

    print(f"\n{ticker} GARCH(1,1) Results:")
    print(f"  omega:  {params['omega']:.6f}")
    print(f"  alpha:  {params['alpha[1]']:.4f}")
    print(f"  beta:   {params['beta[1]']:.4f}")
    print(f"  alpha+beta: {params['alpha[1]'] + params['beta[1]']:.4f}")
    print(f"  VaR 95%: {var_95:.4f}")

    return {
        "ticker": ticker,
        "omega": params["omega"],
        "alpha": params["alpha[1]"],
        "beta": params["beta[1]"],
        "persistence": params["alpha[1]"] + params["beta[1]"],
        "var_95": var_95,
        "cond_vol": cond_vol
    }

if __name__ == "__main__":
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)

    all_vol = {}
    summary_rows = []

    for ticker in returns.columns:
        result = fit_garch(returns[ticker], ticker)
        all_vol[ticker] = result["cond_vol"]
        summary_rows.append({
            "ticker": result["ticker"],
            "omega": result["omega"],
            "alpha": result["alpha"],
            "beta": result["beta"],
            "persistence": result["persistence"],
            "var_95": result["var_95"]
        })

    # Save conditional volatility time series
    vol_df = pd.DataFrame(all_vol)
    vol_df.index = returns.index[len(returns.index) - len(vol_df):] 
    vol_df.to_csv("data/volatility.csv")

    # Save model summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("data/model_summary.csv", index=False)

    print("\nFiles saved: data/volatility.csv, data/model_summary.csv")