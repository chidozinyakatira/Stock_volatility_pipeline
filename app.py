import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Stock Volatility Dashboard",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    vol = pd.read_csv("data/volatility.csv", index_col=0, parse_dates=True)
    summary = pd.read_csv("data/model_summary.csv")
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    spectral = pd.read_csv("data/spectral.csv")
    return vol, summary, returns, spectral

vol, summary, returns, spectral = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("📈 Controls")
    st.divider()
    selected_tickers = st.multiselect(
        "Select stocks:",
        options=vol.columns.tolist(),
        default=vol.columns.tolist()
    )
    st.divider()
    date_range = st.date_input(
        "Date range:",
        value=[vol.index.min(), vol.index.max()],
        min_value=vol.index.min(),
        max_value=vol.index.max()
    )
    st.divider()
    st.caption("Data: Yahoo Finance | Model: GARCH(1,1) + Spectral Analysis")

# --- FILTER ---
if len(date_range) == 2:
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])
    vol_filtered = vol.loc[start:end, selected_tickers]
    ret_filtered = returns.loc[start:end, selected_tickers]
else:
    vol_filtered = vol[selected_tickers]
    ret_filtered = returns[selected_tickers]

filtered_summary = summary[summary["ticker"].isin(selected_tickers)]
filtered_spectral = spectral[spectral["ticker"].isin(selected_tickers)]

# --- HEADER ---
st.title("📈 Stock Volatility Dashboard")
st.caption("GARCH(1,1) conditional volatility + spectral analysis across FAANG stocks -- 2018 to 2024")
st.divider()

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks Selected", len(selected_tickers))

if not filtered_summary.empty:
    most_vol = filtered_summary.loc[filtered_summary["var_95"].idxmin(), "ticker"]
    most_per = filtered_summary.loc[filtered_summary["persistence"].idxmax(), "ticker"]
    avg_var = filtered_summary["var_95"].mean()
    col2.metric("Most Volatile", most_vol)
    col3.metric("Most Persistent", most_per)
    col4.metric("Avg VaR 95%", f"{avg_var:.2%}")

st.divider()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Conditional Volatility",
    "📉 Log Returns",
    "🔢 Model Parameters",
    "📡 Spectral Analysis"
])

with tab1:
    st.subheader("Annualised Conditional Volatility")
    st.caption("Spikes indicate market stress -- notice the COVID crash in early 2020.")
    st.line_chart(vol_filtered)

with tab2:
    st.subheader("Daily Log Returns")
    st.caption("Volatility clustering is visible -- calm periods followed by turbulent ones.")
    st.line_chart(ret_filtered)

with tab3:
    st.subheader("GARCH(1,1) Model Parameters")
    st.caption("Persistence (alpha + beta) close to 1 means shocks take longer to fade.")
    display = filtered_summary.copy()
    display["var_95"] = display["var_95"].apply(lambda x: f"{x:.2%}")
    display["persistence"] = display["persistence"].apply(lambda x: f"{x:.4f}")
    display["alpha"] = display["alpha"].apply(lambda x: f"{x:.4f}")
    display["beta"] = display["beta"].apply(lambda x: f"{x:.4f}")
    display.columns = ["Ticker", "Omega", "Alpha", "Beta", "Persistence", "VaR 95%"]
    st.dataframe(display, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Power Spectral Density")
    st.caption("Peaks reveal dominant volatility cycles. META's ~31-day cycle stands out from the group.")

    # Dominant cycles summary
    st.markdown("**Dominant Cycles per Stock**")
    dominant = (
        filtered_spectral.loc[filtered_spectral.groupby("ticker")["power"].idxmax()]
        [["ticker", "period_days", "frequency", "power"]]
        .copy()
    )
    dominant["period_days"] = dominant["period_days"].apply(lambda x: f"{x:.1f} days")
    dominant["frequency"] = dominant["frequency"].apply(lambda x: f"{x:.4f}")
    dominant["power"] = dominant["power"].apply(lambda x: f"{x:.4f}")
    dominant.columns = ["Ticker", "Dominant Cycle", "Frequency", "Power"]
    st.dataframe(dominant, use_container_width=True, hide_index=True)

    st.divider()

    # PSD chart per stock
    st.markdown("**Periodogram -- Power vs Period (days)**")
    ticker_choice = st.selectbox("Select stock to view periodogram:", selected_tickers)

    stock_spectral = filtered_spectral[
        (filtered_spectral["ticker"] == ticker_choice) &
        (filtered_spectral["period_days"] <= 500)  # cap for readability
    ].copy()

    chart_data = stock_spectral.set_index("period_days")[["power"]]
    st.line_chart(chart_data)
    st.caption("X-axis: cycle length in trading days. Peaks = dominant cycles.")