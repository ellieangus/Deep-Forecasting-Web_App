# Nixtla Time Series Forecasting App

A production-style Streamlit web application for univariate time series forecasting. Compare classical statistics, tree-based machine learning, and deep learning models side-by-side — no code required.

**Live Demo:** [deep-forecasting-webapp.streamlit.app](https://deep-forecasting-webapp-a8lhrxktlz83yawzc6zggi.streamlit.app/)

---

## What It Does

Load a time series (from a built-in dataset or your own CSV), configure your models and horizon in the sidebar, and explore:

- Forecast accuracy vs. a held-out test set
- Rolling backtests across multiple time windows
- Head-to-head model comparisons
- True future projections beyond your last data point

---

## Features

- **3 built-in sample datasets** — AirPassengers, US Macro Monthly, US Macro Quarterly
- **Upload any CSV** with a date column and a numeric target
- **8 forecasting models** across three paradigms (stats, ML, deep learning)
- **5 accuracy metrics** — MAE, RMSE, MAPE, sMAPE, MASE
- **Rolling backtesting** via Nixtla's native `cross_validation`
- **Residuals analysis** — time-series plot, error distribution histogram, summary stats
- **Model Duel** — head-to-head RMSE comparison with winner callout
- **Future Forecast** — train on 100% of data, project h steps into the true future
- **Config save/load** — export and restore sidebar settings as JSON
- **CSV download** for forecast results, backtest data, and future projections

---

## App Tabs

| Tab | What it does |
|-----|-------------|
| **📊 Data** | View your time series, train/test split, and descriptive statistics |
| **🔮 Forecast** | Run models, compare metrics, inspect individual model plots and residuals |
| **🔁 Backtesting** | Rolling cross-validation across multiple cutoff windows for robust accuracy estimates |
| **⚔️ Model Duel** | Pick any two models from your forecast results for a direct head-to-head comparison |
| **🔭 Future Forecast** | Train on 100% of your data and project h steps beyond the last known date |
| **ℹ️ About** | Metric definitions, library overview, and architecture reference |

---

## Models

### statsforecast — Classical Statistics
| Model | Description |
|-------|-------------|
| **AutoARIMA** | Automatically selects the best ARIMA order (p, d, q) and seasonal terms via information criteria |
| **AutoETS** | Automatic Exponential Smoothing — selects Error, Trend, and Seasonality components |
| **SeasonalNaive** | Repeats the last observed seasonal cycle; a strong baseline for seasonal data |

### mlforecast — Tree-Based Machine Learning
These models treat forecasting as a supervised regression problem using lag features and calendar variables.

| Model | Description |
|-------|-------------|
| **LightGBM** | Gradient boosting with fast training and strong accuracy on tabular data |
| **XGBoost** | Gradient boosted trees — robust and widely used in practice |
| **RandomForest** | Ensemble of decision trees; more stable but slower than boosting methods |

### neuralforecast — Deep Learning
Recurrent neural networks that learn temporal patterns directly from the sequence.

| Model | Description |
|-------|-------------|
| **LSTM** | Long Short-Term Memory — captures long-range dependencies via gated memory cells |
| **GRU** | Gated Recurrent Unit — similar to LSTM but with fewer parameters and faster training |

> Neural models are trained with `max_steps` gradient steps (configurable in the sidebar). Lower = faster but less accurate; higher = slower but better fit.

---

## Libraries

| Library | Role |
|---------|------|
| `statsforecast` | Classical statistical models |
| `mlforecast` | ML models with lag/date feature engineering |
| `neuralforecast` | Deep learning models (PyTorch backend) |
| `utilsforecast` | Shared Nixtla utilities |
| `streamlit` | Web app framework |
| `plotly` | Interactive charts |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` / `xgboost` / `lightgbm` | ML estimators |
| `torch` | Neural network backend |

All three Nixtla libraries share the same long-format input schema:

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | string | Series identifier |
| `ds` | datetime | Timestamp |
| `y` | float | Target variable |

`app.py` converts any input into this format automatically.

---

## Architecture

```
Deep-Forecasting-Web_App/
├── app.py                    # Streamlit UI + model orchestration
├── modules/
│   ├── evaluation.py         # compute_metrics() — MAE, RMSE, MAPE, sMAPE, MASE
│   └── config_manager.py     # build_config(), parse_config_bytes()
├── data/
│   ├── airline_passengers.csv
│   ├── US_macro_monthly.csv
│   └── US_macro_Quarterly.csv
├── requirements.txt
├── runtime.txt               # Pins Python 3.11 for Streamlit Cloud
└── .python-version           # uv-compatible Python version pin
```

---

## How to Interpret Results

### Accuracy Metrics

| Metric | Formula | Best for |
|--------|---------|----------|
| **MAE** | mean(\|y − ŷ\|) | Error in original units; easy to explain |
| **RMSE** | √mean((y − ŷ)²) | When large errors should be penalised more |
| **MAPE** | mean(\|y − ŷ\| / y) × 100% | Comparing across different scales |
| **sMAPE** | mean(200\|y − ŷ\| / (\|y\| + \|ŷ\|)) | Like MAPE but symmetric; avoids division-by-zero |
| **MASE** | MAE / seasonal naive MAE | Scale-free; **< 1** means better than a naïve seasonal baseline |

**Lower is always better.** The metrics table highlights the best value per column in blue.

### Residuals

Residual = Actual − Predicted. A good model shows residuals that:
- Hover near **zero** on average (no systematic bias)
- Have no clear **trend or pattern** over time
- Are roughly **symmetrically distributed** in the histogram

### Backtesting vs. Forecast Tab

The **Forecast tab** splits your data once (train/test) and evaluates on that single split. **Backtesting** rolls the training window forward across multiple cutoffs — a more honest and robust accuracy estimate, especially for small datasets.

---

## Uploading Your Own CSV

1. In the sidebar, set **Data source → Upload CSV**
2. Select your file — it must be a `.csv`
3. Choose your **date column** and **target variable** from the dropdowns
4. Set the correct **frequency** (Monthly, Quarterly, Daily, etc.) and **season length**

**CSV requirements:**
- At least **2 columns**: one date/time column and one numeric target column
- At least **20 observations**
- Date column must be parseable by pandas (e.g., `2023-01`, `2023-01-01`, `Q1 2023`)
- Target column must be numeric (integers or floats)
- No requirement on column names — you select them in the sidebar

**Tips:**
- If your data is monthly, set frequency to `Monthly (MS)` and season length to `12`
- If your data is quarterly, use `Quarterly End (Q)` for month-end dates (e.g., `1959-03-31`) or `Quarterly Start (QS)` for month-start dates
- If your series is very short (< 50 points), avoid neural models or reduce `max_steps`

---

## Run Locally

**Requirements:** Python 3.11

```bash
git clone https://github.com/ellieangus/Deep-Forecasting-Web_App.git
cd Deep-Forecasting-Web_App
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> Neural models require PyTorch. If you only want stats/ML models, you can comment out `neuralforecast`, `torch` from `requirements.txt` for a much faster install.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (must be **public**)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — dependencies install automatically

The `runtime.txt` and `.python-version` files pin Python to 3.11, which is required for pre-built wheels of `scipy`, `torch`, and `numba`.

---

## Course Info

Built for DATA 5630 — Deep Forecasting final project (Track 2: Advanced Nixtla App).
