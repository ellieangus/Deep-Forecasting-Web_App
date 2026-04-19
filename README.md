# Nixtla Time Series Forecasting App

A production-style Streamlit web application for univariate time series forecasting,
built on Nixtla's open-source ecosystem.

## Live Demo
<!-- Add your Streamlit Cloud URL here after deploying -->

---

## Features

- **Upload any CSV** or use the built-in AirPassengers demo dataset
- **Multi-model comparison** across three forecasting paradigms:
  - `statsforecast` — AutoARIMA, AutoETS, SeasonalNaive
  - `mlforecast` — LightGBM, XGBoost, RandomForest
  - `neuralforecast` — LSTM, GRU
- **Rolling backtesting** using Nixtla's native `cross_validation` (configurable windows)
- **Combined comparison plots** and per-model visualisations
- **Metrics table** (MAE, RMSE, MAPE) with colour-coded best values
- **Config save / load** — export and restore your model settings as JSON
- **CSV export** for forecast results and backtest data

---

## Architecture

```
app.py                     ← Streamlit UI + forecasting orchestration
modules/
  evaluation.py            ← compute_metrics(), make_leaderboard()
  config_manager.py        ← build_config(), parse_config_bytes()
df_statsforecast.py        ← StatsforecastForecaster wrapper class
df_mlforecast.py           ← MLForecastForecaster wrapper class
df_neuralforecast.py       ← NeuralForecastForecaster wrapper class
airline_passengers.csv     ← Built-in demo dataset
requirements.txt
README.md
```

### How the Nixtla libraries fit together

All three libraries share the same long-format input schema:

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | string | Series identifier |
| `ds` | datetime | Timestamp |
| `y` | float | Target variable |

`app.py` converts any uploaded CSV into this format before passing it to any library.

---

## Run Locally

```bash
git clone https://github.com/ellieangus/Deep-Forecasting-Web_App.git
cd Deep-Forecasting-Web_App
pip install -r requirements.txt
streamlit run app.py
```

Python 3.9+ recommended.

---

## Deploy on Streamlit Cloud

1. Fork / push this repo to GitHub (must be **public**)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — Streamlit Cloud installs dependencies automatically

---

## How to Interpret Results

| Metric | Formula | Use when |
|--------|---------|----------|
| **MAE** | mean(\|y − ŷ\|) | You want error in original units |
| **RMSE** | sqrt(mean((y − ŷ)²)) | Large errors should be penalised more |
| **MAPE** | mean(\|y − ŷ\| / y) × 100 % | Comparing across different scales |

Lower is better for all three. The leaderboard table highlights the best value in green.

**Backtesting** rolls a training window forward `n_windows` times and measures forecast
accuracy at each cutoff — a more honest performance estimate than a single train/test split.

---

## Course Info

Built for the Deep Forecasting final project (Track 2 — Advanced Nixtla App).
