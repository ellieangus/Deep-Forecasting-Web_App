"""
Nixtla Time Series Forecasting App — Track 2 (Advanced)
app.py: Main Streamlit Application
"""

import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.config_manager import build_config, parse_config_bytes
from modules.evaluation import compute_metrics

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Nixtla Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────
# Custom CSS — navy / blue palette
# ──────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #F0F5FB; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1B3A6B; }

.app-header {
    background: linear-gradient(135deg, #1B3A6B 0%, #2E5FA3 100%);
    padding: 1.3rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
}
.app-header h1 { color: white; margin: 0; font-size: 1.7rem; }
.app-header p  { color: #B8D4ED; margin: 0.25rem 0 0; font-size: 0.9rem; }

.winner-box {
    background: linear-gradient(135deg, #1B3A6B 0%, #2E5FA3 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.75rem 0 1.25rem;
}
.winner-box h2 { color: white; margin: 0; font-size: 1.6rem; }
.winner-box p  { color: #B8D4ED; margin: 0.3rem 0 0; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
# Constants & sample dataset registry
# ──────────────────────────────────────────────────
FREQ_MAP = {
    "Monthly (MS)":        "MS",
    "Quarterly End (Q)":   "Q",
    "Quarterly Start (QS)": "QS",
    "Annual (YS)":         "YS",
    "Weekly (W)":          "W",
    "Daily (D)":           "D",
    "Hourly (H)":          "H",
}
STATS_MODELS  = ["AutoARIMA", "AutoETS", "SeasonalNaive"]
ML_MODELS     = ["LightGBM", "XGBoost", "RandomForest"]
NEURAL_MODELS = ["LSTM", "GRU"]

SAMPLE_DATASETS = {
    "AirPassengers": {
        "file":        "data/airline_passengers.csv",
        "date_col":    "Month",
        "target_col":  "Passengers",
        "freq_label":  "Monthly (MS)",
        "season":      12,
    },
    "US Macro Monthly": {
        "file":        "data/US_macro_monthly.csv",
        "date_col":    "DATE",
        "target_col":  "CPI",
        "freq_label":  "Monthly (MS)",
        "season":      12,
    },
    "US Macro Quarterly": {
        "file":        "data/US_macro_Quarterly.csv",
        "date_col":    "Date",
        "target_col":  "cpi",
        "freq_label":  "Quarterly End (Q)",
        "season":      4,
    },
}

# Eight blue shades for multi-model comparison charts (dark → light)
PALETTE = [
    "#1B3A6B", "#1E5FA3", "#2D7DD2", "#4A90D9",
    "#6BAED6", "#9ECAE1", "#B8D4ED", "#0D2347",
]

# Per-family colours — navy / bright blue / light blue for clear contrast
FAMILY_COLORS = {
    "stats":  "#1B3A6B",  # dark navy
    "ml":     "#2D7DD2",  # bright/saturated blue
    "neural": "#B8D4ED",  # very light blue
}

TRAIN_LINE_COLOR = "#6B9FCE"   # slightly darker than lightest blue — visible on white

METRIC_COLS    = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]
METRIC_FMT     = {c: ("{:.2f}%" if c in ("MAPE", "sMAPE") else "{:.3f}") for c in METRIC_COLS}
BT_METRIC_COLS = ["MAE", "RMSE", "MAPE", "sMAPE"]
BT_METRIC_FMT  = {c: METRIC_FMT[c] for c in BT_METRIC_COLS}


def model_color(name: str) -> str:
    """Return the family colour for a given model name."""
    if name in STATS_MODELS:  return FAMILY_COLORS["stats"]
    if name in ML_MODELS:     return FAMILY_COLORS["ml"]
    if name in NEURAL_MODELS: return FAMILY_COLORS["neural"]
    return FAMILY_COLORS["ml"]


# ──────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────
@st.cache_data
def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV from disk, renaming the legacy 'Unnamed: 0' index column to 'Date'."""
    df = pd.read_csv(filepath)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Date"})
    return df


def to_nixtla(df: pd.DataFrame, date_col: str, target_col: str, uid: str = "series_1") -> pd.DataFrame:
    """Convert an arbitrary DataFrame to Nixtla long format (unique_id, ds, y)."""
    out = df[[date_col, target_col]].copy()
    out.columns = ["ds", "y"]
    out["ds"] = pd.to_datetime(out["ds"])
    out["unique_id"] = uid
    out = out[["unique_id", "ds", "y"]].sort_values("ds").reset_index(drop=True)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.dropna(subset=["y"])


# ──────────────────────────────────────────────────
# Model factories
# ──────────────────────────────────────────────────
def make_sf_models(selected, season_length):
    """Instantiate the requested statsforecast model objects."""
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    mapping = {
        "AutoARIMA":     AutoARIMA(season_length=season_length),
        "AutoETS":       AutoETS(season_length=season_length, model="ZZZ"),
        "SeasonalNaive": SeasonalNaive(season_length=season_length),
    }
    return [mapping[m] for m in selected if m in mapping]


def make_ml_dict(selected):
    """Build a dict of {model_name: sklearn estimator} for the selected ML models."""
    from sklearn.ensemble import RandomForestRegressor
    out = {}
    for m in selected:
        if m == "LightGBM":
            try:
                from lightgbm import LGBMRegressor
                out[m] = LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
            except ImportError:
                st.warning("LightGBM not installed — skipping.")
        elif m == "XGBoost":
            try:
                from xgboost import XGBRegressor
                out[m] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            except ImportError:
                st.warning("XGBoost not installed — skipping.")
        elif m == "RandomForest":
            out[m] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return out


def make_nf_models(selected, horizon, input_size, max_steps=50):
    """Instantiate the requested neuralforecast model objects with shared hyperparameters."""
    from neuralforecast.losses.pytorch import MAE as NxMAE
    common = dict(
        input_size=max(input_size, 2),
        loss=NxMAE(),
        max_steps=max_steps,
        scaler_type="robust",
        random_seed=42,
    )
    out = {}
    for m in selected:
        if m == "LSTM":
            from neuralforecast.models import LSTM
            out[m] = LSTM(h=horizon, encoder_hidden_size=16, encoder_n_layers=1,
                          decoder_hidden_size=16, **common)
        elif m == "GRU":
            from neuralforecast.models import GRU
            out[m] = GRU(h=horizon, encoder_hidden_size=16, encoder_n_layers=1,
                         decoder_hidden_size=16, **common)
    return out


# ──────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────
def _date_features(freq):
    """Return calendar date features appropriate for the given frequency."""
    return ["month"] if freq in ("MS", "M", "QS", "Q", "QE", "YS", "Y") else []


# ──────────────────────────────────────────────────
# Forecast runners  (train → test evaluation)
# ──────────────────────────────────────────────────
def run_stats_forecast(train_df, test_df, horizon, freq, season_length, selected):
    """Train statsforecast models on train_df and forecast h steps; evaluate against test_df."""
    from statsforecast import StatsForecast
    models = make_sf_models(selected, season_length)
    if not models:
        return {}
    sf = StatsForecast(models=models, freq=freq, verbose=False)
    preds = sf.forecast(df=train_df, h=horizon)
    out = {}
    for m in selected:
        if m not in preds.columns:
            continue
        merged = preds[["ds", m]].merge(test_df[["ds", "y"]], on="ds", how="left")
        out[m] = {
            "ds":      preds["ds"].values,
            "y_pred":  preds[m].values,
            "metrics": compute_metrics(merged["y"].values, merged[m].values,
                                       y_train=train_df["y"].values),
        }
    return out


def run_ml_forecast(train_df, test_df, horizon, freq, season_length, selected):
    """Train mlforecast models on train_df and forecast h steps; evaluate against test_df."""
    from mlforecast import MLForecast
    ml_dict = make_ml_dict(selected)
    if not ml_dict:
        return {}
    mlf = MLForecast(models=ml_dict, freq=freq,
                     lags=[1, season_length], date_features=_date_features(freq))
    mlf.fit(df=train_df)
    preds = mlf.predict(h=horizon)
    out = {}
    for m in selected:
        if m not in preds.columns:
            continue
        merged = preds[["ds", m]].merge(test_df[["ds", "y"]], on="ds", how="left")
        out[m] = {
            "ds":      preds["ds"].values,
            "y_pred":  preds[m].values,
            "metrics": compute_metrics(merged["y"].values, merged[m].values,
                                       y_train=train_df["y"].values),
        }
    return out


def run_neural_forecast(train_df, test_df, horizon, freq, season_length, selected, max_steps=50):
    """Train neuralforecast models on train_df and forecast h steps; evaluate against test_df."""
    from neuralforecast import NeuralForecast
    input_size = min(2 * season_length, max(len(train_df) // 3, 2))
    nf_dict = make_nf_models(selected, horizon, input_size, max_steps=max_steps)
    if not nf_dict:
        return {}
    nf = NeuralForecast(models=list(nf_dict.values()), freq=freq)
    nf.fit(df=train_df)
    preds = nf.predict(df=train_df)
    out = {}
    for m in selected:
        if m not in preds.columns:
            continue
        merged = preds[["ds", m]].merge(test_df[["ds", "y"]], on="ds", how="left")
        out[m] = {
            "ds":      preds["ds"].values,
            "y_pred":  preds[m].values,
            "metrics": compute_metrics(merged["y"].values, merged[m].values,
                                       y_train=train_df["y"].values),
        }
    return out


# ──────────────────────────────────────────────────
# Future forecast runners  (full data → true future)
# ──────────────────────────────────────────────────
def run_stats_future(full_df, horizon, freq, season_length, selected):
    """Train statsforecast models on the full dataset and forecast h steps into the true future."""
    from statsforecast import StatsForecast
    models = make_sf_models(selected, season_length)
    if not models:
        return {}
    sf = StatsForecast(models=models, freq=freq, verbose=False)
    preds = sf.forecast(df=full_df, h=horizon)
    return {m: {"ds": preds["ds"].values, "y_pred": preds[m].values}
            for m in selected if m in preds.columns}


def run_ml_future(full_df, horizon, freq, season_length, selected):
    """Train mlforecast models on the full dataset and forecast h steps into the true future."""
    from mlforecast import MLForecast
    ml_dict = make_ml_dict(selected)
    if not ml_dict:
        return {}
    mlf = MLForecast(models=ml_dict, freq=freq,
                     lags=[1, season_length], date_features=_date_features(freq))
    mlf.fit(df=full_df)
    preds = mlf.predict(h=horizon)
    return {m: {"ds": preds["ds"].values, "y_pred": preds[m].values}
            for m in selected if m in preds.columns}


def run_neural_future(full_df, horizon, freq, season_length, selected, max_steps=50):
    """Train neuralforecast models on the full dataset and forecast h steps into the true future."""
    from neuralforecast import NeuralForecast
    input_size = min(2 * season_length, max(len(full_df) // 3, 2))
    nf_dict = make_nf_models(selected, horizon, input_size, max_steps=max_steps)
    if not nf_dict:
        return {}
    nf = NeuralForecast(models=list(nf_dict.values()), freq=freq)
    nf.fit(df=full_df)
    preds = nf.predict(df=full_df)
    return {m: {"ds": preds["ds"].values, "y_pred": preds[m].values}
            for m in selected if m in preds.columns}


# ──────────────────────────────────────────────────
# Backtest runners
# ──────────────────────────────────────────────────
def run_stats_backtest(full_df, horizon, freq, season_length, selected, n_windows):
    """Run statsforecast rolling cross-validation over n_windows backtest windows."""
    from statsforecast import StatsForecast
    models = make_sf_models(selected, season_length)
    if not models:
        return {}
    sf = StatsForecast(models=models, freq=freq, verbose=False)
    cv = sf.cross_validation(df=full_df, h=horizon, n_windows=n_windows)
    return {m: cv[["unique_id", "ds", "cutoff", "y", m]].rename(columns={m: "y_pred"})
            for m in selected if m in cv.columns}


def run_ml_backtest(full_df, horizon, freq, season_length, selected, n_windows):
    """Run mlforecast rolling cross-validation over n_windows backtest windows."""
    from mlforecast import MLForecast
    ml_dict = make_ml_dict(selected)
    if not ml_dict:
        return {}
    mlf = MLForecast(models=ml_dict, freq=freq,
                     lags=[1, season_length], date_features=_date_features(freq))
    cv = mlf.cross_validation(df=full_df, h=horizon, n_windows=n_windows)
    return {m: cv[["unique_id", "ds", "cutoff", "y", m]].rename(columns={m: "y_pred"})
            for m in selected if m in cv.columns}


def run_neural_backtest(full_df, horizon, freq, season_length, selected, n_windows, max_steps=30):
    """Run neuralforecast rolling cross-validation over n_windows backtest windows."""
    from neuralforecast import NeuralForecast
    input_size = min(2 * season_length, max(len(full_df) // (n_windows + 3), 2))
    nf_dict = make_nf_models(selected, horizon, input_size, max_steps=max_steps)
    if not nf_dict:
        return {}
    nf = NeuralForecast(models=list(nf_dict.values()), freq=freq)
    cv = nf.cross_validation(df=full_df, h=horizon, n_windows=n_windows)
    return {m: cv[["unique_id", "ds", "cutoff", "y", m]].rename(columns={m: "y_pred"})
            for m in selected if m in cv.columns}


# ──────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

# ── Dataset ─────────────────────────────────────
with st.sidebar.expander("📂 Dataset", expanded=True):
    data_source = st.radio("Data source", ["Sample Dataset", "Upload CSV"],
                           horizontal=True, key="data_source")

    if data_source == "Sample Dataset":
        dataset_name = st.selectbox("Select dataset", list(SAMPLE_DATASETS.keys()))
        meta = SAMPLE_DATASETS[dataset_name]
        raw_df         = load_csv(meta["file"])
        default_date   = meta["date_col"]
        default_target = meta["target_col"]
        default_freq_label = meta["freq_label"]
        default_season = meta["season"]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
                default_date   = raw_df.columns[0]
                default_target = raw_df.columns[1]
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()
        else:
            st.info("Upload a CSV file to get started.")
            st.stop()
        default_freq_label = "Monthly (MS)"
        default_season     = 12

# ── Columns ─────────────────────────────────────
with st.sidebar.expander("📋 Columns", expanded=True):
    all_cols = raw_df.columns.tolist()
    num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found. Please upload a CSV that contains at least one numeric target variable.")
        st.stop()
    if len(all_cols) < 2:
        st.error("CSV must have at least two columns (a date column and a numeric target).")
        st.stop()
    date_col   = st.selectbox("Date column", all_cols,
                               index=all_cols.index(default_date) if default_date in all_cols else 0)
    target_col = st.selectbox("Target variable", num_cols,
                               index=num_cols.index(default_target) if default_target in num_cols else 0)

# ── Time Series Settings ─────────────────────────
with st.sidebar.expander("📅 Time Series Settings", expanded=True):
    freq_keys  = list(FREQ_MAP.keys())
    freq_label = st.selectbox("Frequency", freq_keys,
                               index=freq_keys.index(default_freq_label)
                               if default_freq_label in freq_keys else 0)
    freq          = FREQ_MAP[freq_label]
    season_length = int(st.number_input("Season length", 1, 365, default_season, step=1))
    horizon       = st.slider("Forecast horizon (steps)", 1, 60, 12)
    test_pct      = st.slider("Test set size (%)", 5, 40, 20)

# ── Models ──────────────────────────────────────
with st.sidebar.expander("🧬 Models", expanded=True):
    st.markdown("**statsforecast**")
    sel_stats  = [m for m in STATS_MODELS  if st.checkbox(m, value=(m in ["AutoARIMA", "AutoETS"]),
                                                           key=f"s_{m}")]
    st.markdown("**mlforecast**")
    sel_ml     = [m for m in ML_MODELS     if st.checkbox(m, value=(m == "LightGBM"),
                                                           key=f"m_{m}")]
    st.markdown("**neuralforecast**")
    sel_neural = [m for m in NEURAL_MODELS if st.checkbox(m, value=(m == "LSTM"),
                                                           key=f"n_{m}")]
    neural_steps = int(st.number_input(
        "Neural max_steps (lower = faster)", 10, 500, 50, step=10,
        help="Reduce for quicker demos; increase for better accuracy",
    ))

# ── Config ──────────────────────────────────────
with st.sidebar.expander("💾 Config", expanded=False):
    cfg = build_config(
        freq=freq, season_length=season_length, horizon=horizon, test_pct=test_pct,
        sel_stats=sel_stats, sel_ml=sel_ml, sel_neural=sel_neural,
    )
    st.download_button(
        "⬇️ Save config", json.dumps(cfg, indent=2),
        "forecast_config.json", "application/json",
    )
    cfg_upload = st.file_uploader("⬆️ Load config", type=["json"], key="cfg_up")
    if cfg_upload:
        try:
            loaded_cfg = parse_config_bytes(cfg_upload.read())
            st.success(
                f"Loaded — stats: {loaded_cfg.get('sel_stats', [])}, "
                f"ml: {loaded_cfg.get('sel_ml', [])}, "
                f"neural: {loaded_cfg.get('sel_neural', [])}"
            )
        except Exception as e:
            st.error(f"Invalid config file: {e}")

# ── Sidebar footer ───────────────────────────────
st.sidebar.markdown("---")
st.sidebar.info("▶️ Head to the **Forecast** tab when you're ready to run models.")


# ──────────────────────────────────────────────────
# Prepare data
# ──────────────────────────────────────────────────
try:
    nixtla_df = to_nixtla(raw_df, date_col, target_col)
except Exception as e:
    st.error(f"Cannot parse columns: {e}")
    st.stop()

if len(nixtla_df) < 20:
    st.error("Need at least 20 observations. Check your column selections.")
    st.stop()

n_test   = max(horizon, int(len(nixtla_df) * test_pct / 100))
train_df = nixtla_df.iloc[:-n_test].copy()
test_df  = nixtla_df.iloc[-n_test:].copy()


# ──────────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────────
tab_data, tab_fc, tab_bt, tab_duel, tab_future, tab_about = st.tabs([
    "📊 Data",
    "🔮 Forecast",
    "🔁 Backtesting",
    "⚔️ Model Duel",
    "🔭 Future Forecast",
    "ℹ️ About",
])


# ════════════════════ TAB: DATA ═══════════════════
with tab_data:
    st.markdown("""
    <div class="app-header">
        <h1>📊 Dataset Overview</h1>
        <p>Explore your time series before running any models.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total observations", len(nixtla_df))
    c2.metric("Training points",    len(train_df))
    c3.metric("Test points",        len(test_df))
    c4.metric("Frequency",          freq)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=train_df["ds"], y=train_df["y"], name="Train",
        line=dict(color=TRAIN_LINE_COLOR),
    ))
    fig_ts.add_trace(go.Scatter(
        x=test_df["ds"], y=test_df["y"], name="Test",
        line=dict(color=FAMILY_COLORS["stats"]),
    ))
    fig_ts.update_layout(
        title=f"{target_col} — Train / Test Split",
        xaxis_title="Date", yaxis_title=target_col,
        height=400, template="plotly_white",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("View raw data"):
        st.dataframe(raw_df, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(
        nixtla_df["y"].describe().to_frame().T.style.format("{:.2f}"),
        use_container_width=True,
    )


# ══════════════════ TAB: FORECAST ═════════════════
with tab_fc:
    st.markdown("""
    <div class="app-header">
        <h1>🔮 Forecast</h1>
        <p>Train models on historical data and evaluate on the held-out test set.</p>
    </div>
    """, unsafe_allow_html=True)

    all_sel = sel_stats + sel_ml + sel_neural
    if not all_sel:
        st.warning("Select at least one model in the sidebar to continue.")
    else:
        if st.button("▶️ Run Forecast", type="primary"):
            forecast_results = {}

            if sel_stats:
                with st.spinner(f"Running statsforecast: {sel_stats}…"):
                    try:
                        forecast_results.update(
                            run_stats_forecast(train_df, test_df, horizon, freq, season_length, sel_stats))
                        st.success(f"✅ statsforecast done: {list(sel_stats)}")
                    except Exception as e:
                        st.error(f"statsforecast error: {e}")

            if sel_ml:
                with st.spinner(f"Running mlforecast: {sel_ml}…"):
                    try:
                        forecast_results.update(
                            run_ml_forecast(train_df, test_df, horizon, freq, season_length, sel_ml))
                        st.success(f"✅ mlforecast done: {list(sel_ml)}")
                    except Exception as e:
                        st.error(f"mlforecast error: {e}")

            if sel_neural:
                with st.spinner(f"Training neural models {sel_neural}… (may take 30–90 s)"):
                    try:
                        forecast_results.update(
                            run_neural_forecast(train_df, test_df, horizon, freq, season_length,
                                                sel_neural, max_steps=neural_steps))
                        st.success(f"✅ neuralforecast done: {list(sel_neural)}")
                    except Exception as e:
                        st.error(f"neuralforecast error: {e}")

            st.session_state["fc_res"]   = forecast_results
            st.session_state["fc_train"] = train_df
            st.session_state["fc_test"]  = test_df

        if st.session_state.get("fc_res"):
            fc_res  = st.session_state["fc_res"]
            tr_plot = st.session_state["fc_train"]
            te_plot = st.session_state["fc_test"]

            if not fc_res:
                st.warning("No forecast results. Check for errors above.")
            else:
                # ── Metrics table ───────────────────────────────
                st.subheader("📊 Model Performance")
                mrows = [{"Model": n, **r["metrics"]} for n, r in fc_res.items()]
                mdf = pd.DataFrame(mrows).sort_values("RMSE").reset_index(drop=True)
                present_cols = [c for c in METRIC_COLS if c in mdf.columns]
                st.dataframe(
                    mdf.style
                       .format({c: METRIC_FMT[c] for c in present_cols}, na_rep="—")
                       .highlight_min(subset=present_cols, color="#C9DDEF"),
                    use_container_width=True,
                )

                # ── Combined comparison chart ────────────────────
                st.subheader("📈 All Models — Comparison")
                n_tail = min(40, len(tr_plot))
                fig_all = go.Figure()
                fig_all.add_trace(go.Scatter(
                    x=tr_plot["ds"].iloc[-n_tail:], y=tr_plot["y"].iloc[-n_tail:],
                    name="Train (recent)", line=dict(color=TRAIN_LINE_COLOR, width=1),
                ))
                fig_all.add_trace(go.Scatter(
                    x=te_plot["ds"], y=te_plot["y"],
                    name="Actual", line=dict(color="#1B3A6B", width=2),
                ))
                for i, (mname, r) in enumerate(fc_res.items()):
                    fig_all.add_trace(go.Scatter(
                        x=r["ds"], y=r["y_pred"], name=mname,
                        line=dict(color=PALETTE[i % len(PALETTE)], dash="dash", width=2),
                    ))
                fig_all.update_layout(
                    title="Actual vs All Forecasts",
                    xaxis_title="Date", yaxis_title=target_col,
                    height=450, template="plotly_white",
                )
                st.plotly_chart(fig_all, use_container_width=True)

                # ── Individual model plots ───────────────────────
                st.subheader("🔍 Individual Model Plots")
                for mname, r in fc_res.items():
                    m = r["metrics"]
                    with st.expander(
                        f"{mname}  |  MAE {m['MAE']:.3f}  "
                        f"RMSE {m['RMSE']:.3f}  MAPE {m['MAPE']:.2f}%"
                    ):
                        fig_ind = go.Figure()
                        fig_ind.add_trace(go.Scatter(
                            x=tr_plot["ds"].iloc[-n_tail:], y=tr_plot["y"].iloc[-n_tail:],
                            name="Train", line=dict(color=TRAIN_LINE_COLOR, width=1),
                        ))
                        fig_ind.add_trace(go.Scatter(
                            x=te_plot["ds"], y=te_plot["y"],
                            name="Actual", line=dict(color="#1B3A6B", width=2),
                        ))
                        fig_ind.add_trace(go.Scatter(
                            x=r["ds"], y=r["y_pred"], name=f"{mname} forecast",
                            line=dict(color=model_color(mname), dash="dash", width=2),
                        ))
                        fig_ind.update_layout(height=350, template="plotly_white")
                        st.plotly_chart(fig_ind, use_container_width=True)

                # ── Residuals analysis ───────────────────────────
                st.subheader("📉 Residuals Analysis")
                st.caption("Residual = Actual − Predicted. Values near zero indicate a good fit.")

                resid_map = {}
                for mname, r in fc_res.items():
                    pred_df_r = pd.DataFrame({"ds": r["ds"], "y_pred": r["y_pred"]})
                    merged_r  = pred_df_r.merge(te_plot[["ds", "y"]], on="ds", how="inner")
                    if len(merged_r) > 0:
                        resid_map[mname] = {
                            "ds":  merged_r["ds"].values,
                            "res": (merged_r["y"] - merged_r["y_pred"]).values,
                        }

                if resid_map:
                    fig_resid = go.Figure()
                    # Prominent zero reference line
                    fig_resid.add_hline(y=0, line_dash="dash",
                                        line_color="#1B3A6B", line_width=2)
                    for mname, d in resid_map.items():
                        fig_resid.add_trace(go.Scatter(
                            x=d["ds"], y=d["res"], name=mname,
                            mode="lines+markers",
                            line=dict(color=model_color(mname), width=2),
                            marker=dict(size=5),
                        ))
                    fig_resid.update_layout(
                        title="Residuals Over Time",
                        xaxis_title="Date", yaxis_title="Residual",
                        height=340, template="plotly_white",
                    )
                    st.plotly_chart(fig_resid, use_container_width=True)

                    col_hist, col_summary = st.columns([3, 2])
                    with col_hist:
                        fig_hist = go.Figure()
                        for mname, d in resid_map.items():
                            fig_hist.add_trace(go.Histogram(
                                x=d["res"], name=mname,
                                opacity=0.65,
                                marker_color=model_color(mname),
                                nbinsx=15,
                            ))
                        fig_hist.update_layout(
                            barmode="overlay",
                            title="Error Distribution",
                            xaxis_title="Residual", yaxis_title="Count",
                            height=300, template="plotly_white",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    with col_summary:
                        st.markdown("**Residual Summary**")
                        summary_rows = []
                        for mname, d in resid_map.items():
                            res = d["res"]
                            summary_rows.append({
                                "Model": mname,
                                "Mean":  float(np.mean(res)),
                                "Std":   float(np.std(res)),
                                "Min":   float(np.min(res)),
                                "Max":   float(np.max(res)),
                            })
                        st.dataframe(
                            pd.DataFrame(summary_rows).set_index("Model")
                              .style.format("{:.2f}"),
                            use_container_width=True,
                        )

                st.download_button(
                    "⬇️ Download metrics CSV",
                    mdf.to_csv(index=False),
                    "forecast_metrics.csv", "text/csv",
                )


# ════════════════ TAB: BACKTESTING ════════════════
with tab_bt:
    st.markdown("""
    <div class="app-header">
        <h1>🔁 Backtesting</h1>
        <p>Rolling cross-validation across multiple windows for robust accuracy estimates.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        "Each **window** trains on all data up to a cutoff date, then forecasts "
        f"`h` steps ahead. Aggregating over windows gives a robust accuracy estimate."
    )

    n_windows     = st.slider("Number of backtest windows", 2, 5, 3)
    inc_neural_bt = st.checkbox("Include neural models in backtesting (much slower)", value=False)
    bt_neural = sel_neural if inc_neural_bt else []
    bt_all    = sel_stats + sel_ml + bt_neural

    if not bt_all:
        st.warning("Select at least one model in the sidebar to continue.")
    else:
        if st.button("▶️ Run Backtesting", type="primary"):
            cv_res = {}

            if sel_stats:
                with st.spinner(f"Cross-validating statsforecast: {sel_stats}…"):
                    try:
                        cv_res.update(run_stats_backtest(
                            nixtla_df, horizon, freq, season_length, sel_stats, n_windows))
                        st.success("✅ statsforecast CV done")
                    except Exception as e:
                        st.error(f"statsforecast CV error: {e}")

            if sel_ml:
                with st.spinner(f"Cross-validating mlforecast: {sel_ml}…"):
                    try:
                        cv_res.update(run_ml_backtest(
                            nixtla_df, horizon, freq, season_length, sel_ml, n_windows))
                        st.success("✅ mlforecast CV done")
                    except Exception as e:
                        st.error(f"mlforecast CV error: {e}")

            if bt_neural:
                with st.spinner("Cross-validating neural models… (several minutes)"):
                    try:
                        cv_res.update(run_neural_backtest(
                            nixtla_df, horizon, freq, season_length,
                            bt_neural, n_windows, max_steps=30))
                        st.success("✅ neuralforecast CV done")
                    except Exception as e:
                        st.error(f"neuralforecast CV error: {e}")

            st.session_state["cv_res"] = cv_res

        if st.session_state.get("cv_res"):
            cv_res = st.session_state["cv_res"]

            if not cv_res:
                st.warning("No backtest results. Check for errors above.")
            else:
                agg_rows = [
                    {"Model": mname, **compute_metrics(cdf["y"].values, cdf["y_pred"].values)}
                    for mname, cdf in cv_res.items()
                ]
                agg_df = pd.DataFrame(agg_rows).sort_values("RMSE").reset_index(drop=True)
                bt_present = [c for c in BT_METRIC_COLS if c in agg_df.columns]

                st.subheader("🏆 Model Leaderboard (averaged over all windows)")
                st.dataframe(
                    agg_df[["Model"] + bt_present].style
                           .format({c: BT_METRIC_FMT[c] for c in bt_present}, na_rep="—")
                           .highlight_min(subset=bt_present, color="#C9DDEF"),
                    use_container_width=True,
                )

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(name="MAE",  x=agg_df["Model"], y=agg_df["MAE"],
                                         marker_color=FAMILY_COLORS["stats"]))
                fig_bar.add_trace(go.Bar(name="RMSE", x=agg_df["Model"], y=agg_df["RMSE"],
                                         marker_color=FAMILY_COLORS["ml"]))
                fig_bar.update_layout(
                    barmode="group",
                    title="Model Comparison — MAE & RMSE (backtest avg)",
                    height=350, template="plotly_white",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                st.subheader("Per-Window Details")
                for mname, cdf in cv_res.items():
                    with st.expander(f"{mname} — per cutoff"):
                        cutoffs = cdf["cutoff"].unique()
                        pw_rows = []
                        for cut in cutoffs:
                            g = cdf[cdf["cutoff"] == cut]
                            row = {"Cutoff": str(cut)}
                            row.update(compute_metrics(g["y"].values, g["y_pred"].values))
                            pw_rows.append(row)
                        pw_df = pd.DataFrame(pw_rows)
                        pw_show = [c for c in ["Cutoff"] + bt_present if c in pw_df.columns]
                        st.dataframe(
                            pw_df[pw_show].style.format(
                                {c: BT_METRIC_FMT[c] for c in bt_present if c in pw_df.columns},
                                na_rep="—"
                            ),
                            use_container_width=True,
                        )

                all_cv = pd.concat(
                    [cdf.assign(Model=m) for m, cdf in cv_res.items()], ignore_index=True)
                st.download_button(
                    "⬇️ Download backtest CSV",
                    all_cv.to_csv(index=False),
                    "backtest_results.csv", "text/csv",
                )


# ══════════════════ TAB: MODEL DUEL ═══════════════
with tab_duel:
    st.markdown("""
    <div class="app-header">
        <h1>⚔️ Model Duel</h1>
        <p>Head-to-head comparison between any two models. Run a forecast first to unlock.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("fc_res"):
        st.info("👆 Run a forecast in the **Forecast** tab first — the duel uses those results.")
    else:
        fc_duel   = st.session_state["fc_res"]
        available = list(fc_duel.keys())

        if len(available) < 2:
            st.warning("You need at least 2 models in your forecast results. "
                       "Go back to the Forecast tab and select more models.")
        else:
            col_pick_a, col_pick_b = st.columns(2)
            with col_pick_a:
                model_a = st.selectbox("Model A", available, index=0, key="duel_a")
            with col_pick_b:
                opts_b  = [m for m in available if m != model_a]
                model_b = st.selectbox("Model B", opts_b, index=0, key="duel_b")

            ra   = fc_duel[model_a]
            rb   = fc_duel[model_b]
            tr_d = st.session_state["fc_train"]
            te_d = st.session_state["fc_test"]

            rmse_a   = ra["metrics"]["RMSE"]
            rmse_b   = rb["metrics"]["RMSE"]
            winner   = model_a if rmse_a <= rmse_b else model_b
            loser    = model_b if rmse_a <= rmse_b else model_a
            win_rmse = min(rmse_a, rmse_b)
            los_rmse = max(rmse_a, rmse_b)
            pct_gap  = (los_rmse - win_rmse) / los_rmse * 100

            st.markdown(f"""
            <div class="winner-box">
                <h2>🏆 {winner} wins</h2>
                <p>Lower RMSE by {pct_gap:.1f}% &nbsp;·&nbsp;
                   {winner}: {win_rmse:.3f} &nbsp;vs&nbsp; {loser}: {los_rmse:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Metrics Comparison")
            cmp_df = pd.DataFrame({
                "Metric": METRIC_COLS,
                model_a:  [ra["metrics"].get(k, float("nan")) for k in METRIC_COLS],
                model_b:  [rb["metrics"].get(k, float("nan")) for k in METRIC_COLS],
            }).set_index("Metric")
            st.dataframe(cmp_df.style.format("{:.3f}", na_rep="—"), use_container_width=True)

            st.subheader("📈 Head-to-Head Plots")
            n_tail = min(40, len(tr_d))
            col_a_plot, col_b_plot = st.columns(2)

            for col, mname, r in [(col_a_plot, model_a, ra), (col_b_plot, model_b, rb)]:
                with col:
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Scatter(
                        x=tr_d["ds"].iloc[-n_tail:], y=tr_d["y"].iloc[-n_tail:],
                        name="Train", line=dict(color=TRAIN_LINE_COLOR, width=1),
                    ))
                    fig_d.add_trace(go.Scatter(
                        x=te_d["ds"], y=te_d["y"],
                        name="Actual", line=dict(color="#1B3A6B", width=2),
                    ))
                    fig_d.add_trace(go.Scatter(
                        x=r["ds"], y=r["y_pred"], name=mname,
                        line=dict(color=model_color(mname), dash="dash", width=2),
                    ))
                    crown = "🏆 " if mname == winner else ""
                    fig_d.update_layout(
                        title=f"{crown}{mname}  |  RMSE {r['metrics']['RMSE']:.3f}",
                        height=360, template="plotly_white",
                    )
                    st.plotly_chart(fig_d, use_container_width=True)


# ════════════════ TAB: FUTURE FORECAST ════════════
with tab_future:
    st.markdown("""
    <div class="app-header">
        <h1>🔭 Future Forecast</h1>
        <p>Train on all available data and project <em>h</em> steps beyond the last known date.</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Unlike the Forecast tab (which evaluates accuracy on held-out data), this trains on "
        "100% of your data and generates true future predictions — no actuals to compare against."
    )

    all_sel_fut = sel_stats + sel_ml + sel_neural
    if not all_sel_fut:
        st.warning("Select at least one model in the sidebar to continue.")
    else:
        if st.button("▶️ Run Future Forecast", type="primary"):
            future_results = {}

            if sel_stats:
                with st.spinner(f"Running statsforecast future: {sel_stats}…"):
                    try:
                        future_results.update(
                            run_stats_future(nixtla_df, horizon, freq, season_length, sel_stats))
                        st.success("✅ statsforecast done")
                    except Exception as e:
                        st.error(f"statsforecast error: {e}")

            if sel_ml:
                with st.spinner(f"Running mlforecast future: {sel_ml}…"):
                    try:
                        future_results.update(
                            run_ml_future(nixtla_df, horizon, freq, season_length, sel_ml))
                        st.success("✅ mlforecast done")
                    except Exception as e:
                        st.error(f"mlforecast error: {e}")

            if sel_neural:
                with st.spinner("Training neural models… (may take 30–90 s)"):
                    try:
                        future_results.update(
                            run_neural_future(nixtla_df, horizon, freq, season_length,
                                              sel_neural, max_steps=neural_steps))
                        st.success("✅ neuralforecast done")
                    except Exception as e:
                        st.error(f"neuralforecast error: {e}")

            st.session_state["future_res"]  = future_results
            st.session_state["future_full"] = nixtla_df

        if st.session_state.get("future_res"):
            fut_res   = st.session_state["future_res"]
            full_hist = st.session_state["future_full"]

            if not fut_res:
                st.warning("No future forecast results. Check for errors above.")
            else:
                last_date = full_hist["ds"].max()
                n_hist    = min(60, len(full_hist))

                fig_fut = go.Figure()
                fig_fut.add_trace(go.Scatter(
                    x=full_hist["ds"].iloc[-n_hist:],
                    y=full_hist["y"].iloc[-n_hist:],
                    name="Historical",
                    line=dict(color="#1B3A6B", width=2),
                ))
                # Use string to avoid Plotly/pandas Timestamp compatibility issue
                fig_fut.add_vline(
                    x=str(last_date),
                    line_dash="dash",
                    line_color=TRAIN_LINE_COLOR,
                    annotation_text="Forecast start",
                    annotation_position="top left",
                    annotation_font_color="#1B3A6B",
                )
                for i, (mname, r) in enumerate(fut_res.items()):
                    fig_fut.add_trace(go.Scatter(
                        x=r["ds"], y=r["y_pred"], name=mname,
                        line=dict(color=PALETTE[i % len(PALETTE)], dash="dash", width=2),
                    ))
                fig_fut.update_layout(
                    title=f"Future Forecast — {horizon} steps beyond {str(last_date.date())}",
                    xaxis_title="Date", yaxis_title=target_col,
                    height=480, template="plotly_white",
                )
                st.plotly_chart(fig_fut, use_container_width=True)

                fut_dfs = []
                for mname, r in fut_res.items():
                    tmp = pd.DataFrame({"ds": r["ds"], mname: r["y_pred"]})
                    fut_dfs.append(tmp.set_index("ds"))

                if fut_dfs:
                    fut_table = pd.concat(fut_dfs, axis=1).reset_index()
                    with st.expander("View future forecast values"):
                        num_cols_fut = [c for c in fut_table.columns if c != "ds"]
                        st.dataframe(
                            fut_table.style.format({c: "{:.2f}" for c in num_cols_fut}),
                            use_container_width=True,
                        )
                    st.download_button(
                        "⬇️ Download future forecast CSV",
                        fut_table.to_csv(index=False),
                        "future_forecast.csv", "text/csv",
                    )


# ══════════════════ TAB: ABOUT ════════════════════
with tab_about:
    st.markdown("""
    <div class="app-header">
        <h1>ℹ️ About This App</h1>
        <p>Architecture, tab guide, and metric definitions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
## Overview

A **production-style time series forecasting tool** built on
[Nixtla's](https://nixtla.io) open-source ecosystem. Choose from three built-in datasets
or upload your own CSV, select models from three paradigms, and explore forecasts,
backtests, head-to-head comparisons, and true future projections.

---

## Sample Datasets

| Dataset | Frequency | Default Target | Observations |
|---------|-----------|---------------|--------------|
| **AirPassengers** | Monthly | Passengers | 144 |
| **US Macro Monthly** | Monthly | CPI | 383 |
| **US Macro Quarterly** | Quarterly | CPI | 203 |

---

## Nixtla Libraries

| Library | Paradigm | Models |
|---------|----------|--------|
| **statsforecast** | Classical statistics | AutoARIMA, AutoETS, SeasonalNaive |
| **mlforecast** | Tree-based ML | LightGBM, XGBoost, RandomForest |
| **neuralforecast** | Deep learning | LSTM, GRU |

All three share the same long-format schema: `unique_id`, `ds`, `y`.

---

## App Tabs

| Tab | Purpose |
|-----|---------|
| 📊 Data | Explore dataset, view train/test split |
| 🔮 Forecast | Run models, compare accuracy, inspect residuals |
| 🔁 Backtesting | Rolling cross-validation across multiple windows |
| ⚔️ Model Duel | Head-to-head RMSE comparison between any two models |
| 🔭 Future Forecast | Train on 100% of data, project true future values |

---

## Architecture

```
app.py                   ← Streamlit UI + orchestration
modules/
  evaluation.py          ← compute_metrics() — MAE, RMSE, MAPE, sMAPE, MASE
  config_manager.py      ← build_config(), parse_config_bytes()
data/
  airline_passengers.csv
  US_macro_monthly.csv
  US_macro_Quarterly.csv
df_statsforecast.py      ← StatsforecastForecaster wrapper
df_mlforecast.py         ← MLForecastForecaster wrapper
df_neuralforecast.py     ← NeuralForecastForecaster wrapper
```

---

## Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\\|y − ŷ\\|) | Error in original units |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more |
| **MAPE** | mean(\\|y − ŷ\\| / y) × 100% | Scale-free percentage |
| **sMAPE** | mean(200\\|y − ŷ\\| / (\\|y\\| + \\|ŷ\\|)) | Symmetric, bounded |
| **MASE** | MAE / naive MAE | < 1 means better than naïve seasonal |

**Lower is always better.**

---

## Run Locally

```bash
git clone https://github.com/ellieangus/Deep-Forecasting-Web_App.git
cd Deep-Forecasting-Web_App
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push repo to GitHub (public)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. New app → connect repo → Main file: `app.py`
4. Click **Deploy**
""")
