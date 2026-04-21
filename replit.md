# Quantum Algo — Quantitative Engine

## Overview
A fully-featured market analysis and price prediction dashboard for financial assets (crypto, stocks, Indian NSE, indices, commodities). Provides live market data, technical indicators, historical candlestick charting, AI-based price forecasting, confluence-based trading signals, news sentiment, portfolio tracking, and multi-asset comparison.

## Tech Stack
- **Backend:** Flask (Python 3.12) + Flask-CORS
- **Market Data:** yfinance (Yahoo Finance)
- **ML/Forecasting:** Prophet, ARIMA (pmdarima), LSTM (TensorFlow/Keras), scikit-learn
- **Technical Indicators:** Computed manually (RSI, MACD, Bollinger Bands)
- **News:** feedparser (Yahoo Finance RSS)
- **Frontend:** HTML5 + Bootstrap 5 + Chart.js + Lightweight Charts v4 (candlesticks)
- **Production Server:** Gunicorn

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | / | Dashboard HTML |
| POST | /api/market-data | Live OHLCV + RSI + MACD + BB + signal |
| GET | /api/model-metrics | Model accuracy comparison |
| POST | /predict/prophet | Prophet forecast |
| POST | /predict/arima | ARIMA forecast |
| POST | /predict/lstm | LSTM recursive forecast |
| GET | /api/news?ticker= | Yahoo Finance RSS + sentiment |
| POST | /api/compare | Multi-ticker normalized % returns |
| POST | /api/portfolio/add | Add portfolio position |
| GET | /api/portfolio | Get all holdings with P&L |
| DELETE | /api/portfolio/remove | Remove holding |

## Frontend Tabs (Top navigation — glassmorphism dark theme)
1. **Command Center** — 8 stat cards: Price, Volume, Trend, 52W High/Low, Market Cap, RSI, MACD Signal. Scrolling ticker tape at top.
2. **Market Analysis** — TradingView Lightweight Charts v4 candlestick + volume + RSI panel + MACD panel + Bollinger Bands overlay toggle + 1W/1M/3M range buttons
3. **Model Comparison** — Table with accuracy bars and Best Model badge
4. **Predictive Engine** — Prophet / ARIMA / LSTM forecast + confidence band + forecast table + CSV export
5. **Strategy & Signals** — Confluence signal (BUY/SELL/HOLD) + per-indicator breakdown table + score bar
6. **News & Sentiment** — Yahoo Finance RSS feed with keyword sentiment + overall sentiment gauge
7. **Watchlist & Alerts** — Add tickers with above/below price alert thresholds; browser notifications via Notification API; live refresh
8. **My Portfolio** — Add/remove positions, doughnut allocation chart, P&L table, totals
9. **Asset Comparison** — Normalized % returns chart for multiple tickers

## Key Features
- In-memory caching: market data 60s, news 5min; X-Cache header
- Auto-refresh every 30s (toggle)
- Toast notification system (replaces alert())
- Skeleton loading placeholders
- Global loading bar
- Mobile responsive with hamburger sidebar
- Custom ticker input (overrides dropdown)
- Download forecast as CSV

## Running
- **Dev:** `python app.py` (Flask on 0.0.0.0:5000)
- **Production:** `gunicorn --bind=0.0.0.0:5000 --reuse-port app:app`

## Model Files
- `arima_model.pkl` — ARIMA trained model
- `prophet_model.pkl` — Prophet trained model
- `lstm_model.h5` — LSTM deep learning model
- `minmax_scaler.pkl` — MinMaxScaler for LSTM normalization (saved with sklearn 1.6.1)
