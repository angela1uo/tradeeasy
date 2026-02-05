import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_stock.csv"


def _try_import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    return yf


@st.cache_data(show_spinner=False)
def load_history(
    ticker: str, start: dt.date, end: dt.date
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    yf = _try_import_yfinance()
    if yf is None:
        return None, "yfinance not installed; using demo data instead."

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - guardrail for network issues
        return None, f"Could not download data: {exc}"

    if df is None or df.empty:
        return None, "No data returned for that range; using demo data instead."

    df = df.reset_index()
    return df, None


def load_uploaded_data(file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = pd.read_csv(file, parse_dates=["Date"])
    except Exception as exc:
        return None, f"Could not read uploaded CSV: {exc}"

    required_cols = {"Date", "Close"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        return None, f"Uploaded CSV missing columns: {', '.join(missing)}"

    if "Open" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
        # Ensure columns exist for preview consistency
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = df["Close"]

    return df.sort_values("Date").reset_index(drop=True), None


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_CSV, parse_dates=["Date"])
    return df


def build_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    prices = df.copy()
    prices["return"] = prices["Close"].pct_change()
    for i in range(1, n_lags + 1):
        prices[f"lag_{i}"] = prices["return"].shift(i)
    prices = prices.dropna().reset_index(drop=True)
    return prices


def train_model(
    df: pd.DataFrame, n_lags: int = 5, forecast_horizon: int = 5
) -> Tuple[LinearRegression, pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    features = build_features(df, n_lags=n_lags)
    X = features[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = features["return"]

    split = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Multi-step forecast via iterative one-step predictions
    recent_returns = list(y.values[-n_lags:])
    forecast_returns = []
    for _ in range(forecast_horizon):
        x_input = np.array(recent_returns[-n_lags:]).reshape(1, -1)
        next_return = model.predict(x_input)[0]
        forecast_returns.append(next_return)
        recent_returns.append(next_return)

    last_close = df["Close"].iloc[-1]
    forecast_prices = [last_close]
    for r in forecast_returns:
        forecast_prices.append(forecast_prices[-1] * (1 + r))

    return model, features, y_test, preds, np.array(forecast_prices[1:])


def plot_history_and_forecast(
    df: pd.DataFrame, forecast_prices: np.ndarray, forecast_horizon: int
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#1f77b4"),
        )
    )

    future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast_prices,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Historical Prices and Forecast",
        yaxis_title="Price",
    )
    return fig


def metrics_section(y_true: pd.Series, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    st.metric("MAE (test)", f"{mae:.4f}")
    st.metric("RMSE (test)", f"{rmse:.4f}")


def sidebar_controls():
    st.sidebar.header("Controls")
    st.sidebar.markdown("**Data source**")
    uploaded = st.sidebar.file_uploader("Upload CSV (Date, Close, ...)", type=["csv"])
    ticker = st.sidebar.text_input("Ticker (if not uploading)", value="AAPL")
    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 2)
    start = st.sidebar.date_input("Start date", value=default_start)
    end = st.sidebar.date_input("End date", value=today)
    n_lags = st.sidebar.slider("Lag features", min_value=3, max_value=10, value=5)
    horizon = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=30, value=7)
    return uploaded, ticker, start, end, n_lags, horizon


def main():
    st.set_page_config(page_title="Stock Educator Dashboard", layout="wide")
    st.title("Interactive Stock Dashboard")
    st.caption("Explore prices, simple forecasts, and model diagnostics.")

    uploaded, ticker, start, end, n_lags, horizon = sidebar_controls()

    data_status = st.empty()
    df, error = (None, None)
    if uploaded is not None:
        df, error = load_uploaded_data(uploaded)
        if df is not None:
            data_status.success(f"Loaded {len(df)} rows from upload.")
    if df is None:
        df, error = load_history(ticker, start, end)
        if df is not None:
            data_status.success(f"Loaded {len(df)} rows for {ticker}.")
    if df is None:
        data_status.warning(error or "Falling back to demo data.")
        df = load_sample_data()
        df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]
        if df.empty:
            st.error("No demo data available for that range. Try widening the dates.")
            return

    st.subheader("Price Overview")
    st.line_chart(df.set_index("Date")["Close"])

    with st.spinner("Training model and forecasting..."):
        model, features, y_test, preds, forecast_prices = train_model(
            df, n_lags=n_lags, forecast_horizon=horizon
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test Metrics")
        metrics_section(y_test, preds)
    with col2:
        st.subheader("Feature Coefficients")
        coef_df = pd.DataFrame(
            {
                "Feature": [f"lag_{i}" for i in range(1, n_lags + 1)],
                "Weight": model.coef_,
            }
        )
        st.bar_chart(coef_df.set_index("Feature"))

    st.subheader("Forecast")
    fig = plot_history_and_forecast(df, forecast_prices, horizon)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(df.tail(20))

    st.markdown(
        """
        **How it works**
        - Fetches daily close data with `yfinance` (or uses demo data if offline).
        - Builds lagged return features.
        - Trains a simple linear regression to predict next-day returns.
        - Rolls predictions forward to estimate future prices.
        """
    )


if __name__ == "__main__":
    main()
