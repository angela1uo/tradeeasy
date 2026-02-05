# TradeEasy – Stock Learning Dashboard

An interactive Streamlit dashboard to explore stock prices, teach basic time-series concepts, and generate a lightweight lag-based forecast.

## Quick start
1. Install dependencies (prefer a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in your browser.

## Features
- Download historical prices with `yfinance` (falls back to included demo data when offline).
- Or upload your own CSV (columns: at least `Date`, `Close`; optional `Open`, `High`, `Low`, `Volume`).
- Adjustable lags and forecast horizon for a simple regression-based predictor.
- Interactive charts with Plotly and quick metrics (MAE/RMSE).
- Demo CSV at `data/sample_stock.csv` so the app works without internet.

## Notes
- If `yfinance` is not installed or internet is unavailable, the app automatically switches to the bundled demo data.
- When uploading your own CSV, ensure the `Date` column is parseable; the app sorts by `Date`.
- The model is intentionally simple (linear regression on lagged returns) to keep the experience educational and fast to run locally. Replace it with a richer model (Prophet, ARIMA, LSTM, etc.) if you want more sophistication.