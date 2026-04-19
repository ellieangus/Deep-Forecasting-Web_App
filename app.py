"""
Nixtla Time Series Forecasting App — Track 2 (Advanced)
app.py: Main Streamlit Application

Libraries used:
  statsforecast  — classical statistical models (AutoARIMA, AutoETS, SeasonalNaive)
  mlforecast     — ML tree-based models (LightGBM, XGBoost, RandomForest)
  neuralforecast — deep learning models (LSTM, GRU)
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
# Constants
# ──────────────────────────────────────────────────
FREQ_MAP = {
    "Monthly (MS)":    "MS",
    "Quarterly (QS)":  "QS",
    "Annual (YS)":     "YS",
    "Weekly (W)":      "W",
    "Daily (D)":       "D",
    "Hourly (H)":      "H",
}
STATS_MODELS  = ["AutoARIMA", "AutoETS", "SeasonalNaive"]
ML_MODELS     = ["LightGBM", "XGBoost", "RandomForest"]
NEURAL_MODELS = ["LSTM", "GRU"]
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# ──────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────

@st.cache_data
def load_airline():
    df = pd.read_csv("airline_passengers.csv")
    df.columns = ["Month", "Passengers"]
    return df


def to_nixtla(df: pd.DataFrame, date_col: str, target_col: str, uid: str = "series_1") -> pd.DataFrame:
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
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    mapping = {
        "AutoARIMA":     AutoARIMA(season_length=season_length),
        "AutoETS":       AutoETS(season_length=season_length, model="ZZZ"),
        "SeasonalNaive": SeasonalNaive(season_length=season_length),
    }
    return [mapping[m] for m in selected if m in mapping]


def make_ml_dict(selected):
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
            out[m] = LSTM(
                h=horizon, encoder_hidden_size=16, encoder_n_layers=1,
                decoder_hidden_size=16, **common,
            )
        elif m == "GRU":
            from neuralforecast.models import GRU
            out[m] = GRU(
                h=horizon, encoder_hidden_size=16, encoder_n_layers=1,
                decoder_hidden_size=16, **common,
            )
    return out


# ──────────────────────────────────────────────────
# Forecast runners
# ──────────────────────────────────────────────────

def _date_features(freq):
    return ["month"] if freq in ("MS", "M", "QS", "Q", "YS", "Y") else []


def run_stats_forecast(train_df, test_df, horizon, freq, season_length, selected):
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
            "ds":     preds["ds"].values,
            "y_pred": preds[m].values,
            "metrics": compute_metrics(merged["y"].values, merged[m].values),
        }
    return out


def run_ml_forecast(train_df, test_df, horizon, freq, season_length, selected):
    from mlforecast import MLForecast
    ml_dict = make_ml_dict(selected)
    if not ml_dict:
        return {}
    mlf = MLForecast(
        models=ml_dict, freq=freq,
        lags=[1, season_length],
        date_features=_date_features(freq),
    )
    mlf.fit(df=train_df)
    preds = mlf.predict(h=horizon)
    out = {}
    for m in selected:
        if m not in preds.columns:
            continue
        merged = preds[["ds", m]].merge(test_df[["ds", "y"]], on="ds", how="left")
        out[m] = {
            "ds":     preds["ds"].values,
            "y_pred": preds[m].values,
            "metrics": compute_metrics(merged["y"].values, merged[m].values),
        }
    return out


def run_neural_forecast(train_df, test_df, horizon, freq, season_length, selected, max_steps=50):
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
            "ds":     preds["ds"].values,
            "y_pred": preds[m].values,
            "metrics": compute_metrics(merged["y"].values, merged[m].values),
        }
    return out


# ──────────────────────────────────────────────────
# Backtest runners
# ──────────────────────────────────────────────────

def run_stats_backtest(full_df, horizon, freq, season_length, selected, n_windows):
    from statsforecast import StatsForecast
    models = make_sf_models(selected, season_length)
    if not models:
        return {}
    sf = StatsForecast(models=models, freq=freq, verbose=False)
    cv = sf.cross_validation(df=full_df, h=horizon, n_windows=n_windows)
    return {
        m: cv[["unique_id", "ds", "cutoff", "y", m]].rename(columns={m: "y_pred"})
        for m in selected
        if m in cv.columns
    }


def run_ml_backtest(full_df, horizon, freq, season_length, selected, n_windows):
    from mlforecast import MLForecast
    ml_dict = make_ml_dict(selected)
    if not ml_dict:
        return {}
    mlf = MLForecast(
        models=ml_dict, freq=freq,
        lags=[1, season_length],
        date_features=_date_features(freq),
    )
    cv = mlf.cross_validation(df=full_df, h=horizon, n_windows=n_windows)
    return {
        m: cv[["unique_id", "ds", "cutoff", "y", m]].rename(columns={m: "y_pred"})
        for m in selected
        if m in cv.columns
    }


def run_neural_backtest(full_df, horizon, freq, season_length, selected, n_windows, max_steps=30):
    from neuralforecast import NeuralForecast
    input_size = min(2 * season_length, max(len(full_df) // (n_windows + 3), 2))
    nf_dict = make_nf_models(selected, horizon, input_size, max_steps=max_steps)
    if not nf_dict:
        return {}
    nf = NeuralForecast(models=list(nf_dict.values()), freq=freq)
    cv = nf.cross_validation(df=full_df, h=horizon, n_windows=n_windows)
    return {
        m: cv[["unique_id", "ds", "cutoff", "y", m]].rename(columns={m: "y_pred"})
        for m in selected
        if m in cv.columns
    }


# ──────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

with st.sidebar.expander("📂 Dataset", expanded=True):
    use_sample = st.checkbox("Use built-in sample (AirPassengers)", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

if use_sample:
    raw_df = load_airline()
    default_date, default_target = "Month", "Passengers"
elif uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        default_date   = raw_df.columns[0]
        default_target = raw_df.columns[1]
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.sidebar.info("Upload a CSV or use the sample dataset.")
    st.stop()

with st.sidebar.expander("🗂 Columns", expanded=True):
    all_cols  = raw_df.columns.tolist()
    num_cols  = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    date_col  = st.selectbox("Date column",   all_cols,
                              index=all_cols.index(default_date) if default_date in all_cols else 0)
    target_col = st.selectbox("Target variable", num_cols,
                               index=num_cols.index(default_target) if default_target in num_cols else 0)

with st.sidebar.expander("📅 Time Series Settings", expanded=True):
    freq_label    = st.selectbox("Frequency", list(FREQ_MAP.keys()), index=0)
    freq          = FREQ_MAP[freq_label]
    season_length = int(st.number_input("Season length", 1, 365, 12, step=1))
    horizon       = st.slider("Forecast horizon (steps)", 1, 60, 12)
    test_pct      = st.slider("Test set size (%)", 5, 40, 20)

with st.sidebar.expander("🤖 Models", expanded=True):
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
tab_data, tab_fc, tab_bt, tab_about = st.tabs([
    "📊 Data", "🔮 Forecast", "🔁 Backtesting", "ℹ️ About",
])


# ════════════════════ TAB: DATA ═══════════════════
with tab_data:
    st.title("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total observations", len(nixtla_df))
    c2.metric("Training points",    len(train_df))
    c3.metric("Test points",        len(test_df))
    c4.metric("Frequency",          freq)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=train_df["ds"], y=train_df["y"], name="Train",
        line=dict(color="#1f77b4"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=test_df["ds"], y=test_df["y"], name="Test",
        line=dict(color="#ff7f0e"),
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
    st.title("Forecast")

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
                            run_stats_forecast(
                                train_df, test_df, horizon, freq, season_length, sel_stats,
                            )
                        )
                        st.success(f"✅ statsforecast done: {list(sel_stats)}")
                    except Exception as e:
                        st.error(f"statsforecast error: {e}")

            if sel_ml:
                with st.spinner(f"Running mlforecast: {sel_ml}…"):
                    try:
                        forecast_results.update(
                            run_ml_forecast(
                                train_df, test_df, horizon, freq, season_length, sel_ml,
                            )
                        )
                        st.success(f"✅ mlforecast done: {list(sel_ml)}")
                    except Exception as e:
                        st.error(f"mlforecast error: {e}")

            if sel_neural:
                with st.spinner(f"Training neural models {sel_neural}… (may take 30–90 s)"):
                    try:
                        forecast_results.update(
                            run_neural_forecast(
                                train_df, test_df, horizon, freq, season_length,
                                sel_neural, max_steps=neural_steps,
                            )
                        )
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
                st.warning("No forecast results to display. Check for errors above.")
            else:
                # Metrics table
                st.subheader("📊 Model Performance")
                mrows = [{"Model": n, **r["metrics"]} for n, r in fc_res.items()]
                mdf = pd.DataFrame(mrows).sort_values("RMSE").reset_index(drop=True)
                st.dataframe(
                    mdf.style
                       .format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}%"})
                       .highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#c3e6cb"),
                    use_container_width=True,
                )

                # Combined comparison chart
                st.subheader("📈 All Models — Comparison")
                n_tail = min(40, len(tr_plot))
                fig_all = go.Figure()
                fig_all.add_trace(go.Scatter(
                    x=tr_plot["ds"].iloc[-n_tail:], y=tr_plot["y"].iloc[-n_tail:],
                    name="Train (recent)", line=dict(color="#aaaaaa", width=1),
                ))
                fig_all.add_trace(go.Scatter(
                    x=te_plot["ds"], y=te_plot["y"],
                    name="Actual", line=dict(color="black", width=2),
                ))
                for i, (mname, r) in enumerate(fc_res.items()):
                    fig_all.add_trace(go.Scatter(
                        x=r["ds"], y=r["y_pred"],
                        name=mname,
                        line=dict(color=PALETTE[i % len(PALETTE)], dash="dash", width=2),
                    ))
                fig_all.update_layout(
                    title="Actual vs All Forecasts",
                    xaxis_title="Date", yaxis_title=target_col,
                    height=450, template="plotly_white",
                )
                st.plotly_chart(fig_all, use_container_width=True)

                # Individual model plots
                st.subheader("🔍 Individual Model Plots")
                for i, (mname, r) in enumerate(fc_res.items()):
                    m = r["metrics"]
                    with st.expander(
                        f"{mname}  |  MAE {m['MAE']:.3f}  "
                        f"RMSE {m['RMSE']:.3f}  MAPE {m['MAPE']:.2f}%"
                    ):
                        fig_ind = go.Figure()
                        fig_ind.add_trace(go.Scatter(
                            x=tr_plot["ds"].iloc[-n_tail:], y=tr_plot["y"].iloc[-n_tail:],
                            name="Train", line=dict(color="#aaaaaa", width=1),
                        ))
                        fig_ind.add_trace(go.Scatter(
                            x=te_plot["ds"], y=te_plot["y"],
                            name="Actual", line=dict(color="black", width=2),
                        ))
                        fig_ind.add_trace(go.Scatter(
                            x=r["ds"], y=r["y_pred"],
                            name=f"{mname} forecast",
                            line=dict(color=PALETTE[i % len(PALETTE)], dash="dash", width=2),
                        ))
                        fig_ind.update_layout(height=350, template="plotly_white")
                        st.plotly_chart(fig_ind, use_container_width=True)

                st.download_button(
                    "⬇️ Download metrics CSV",
                    mdf.to_csv(index=False),
                    "forecast_metrics.csv", "text/csv",
                )


# ════════════════ TAB: BACKTESTING ════════════════
with tab_bt:
    st.title("Backtesting — Rolling Cross-Validation")
    st.markdown(
        "Each **window** trains on all data up to a cutoff date, then forecasts "
        f"`h` steps ahead. Aggregating over windows gives a robust accuracy estimate."
    )

    n_windows     = st.slider("Number of backtest windows", 2, 5, 3)
    inc_neural_bt = st.checkbox(
        "Include neural models in backtesting (much slower)", value=False,
    )
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
                        cv_res.update(
                            run_stats_backtest(
                                nixtla_df, horizon, freq, season_length, sel_stats, n_windows,
                            )
                        )
                        st.success(f"✅ statsforecast CV done")
                    except Exception as e:
                        st.error(f"statsforecast CV error: {e}")

            if sel_ml:
                with st.spinner(f"Cross-validating mlforecast: {sel_ml}…"):
                    try:
                        cv_res.update(
                            run_ml_backtest(
                                nixtla_df, horizon, freq, season_length, sel_ml, n_windows,
                            )
                        )
                        st.success(f"✅ mlforecast CV done")
                    except Exception as e:
                        st.error(f"mlforecast CV error: {e}")

            if bt_neural:
                with st.spinner(f"Cross-validating neural models… (several minutes)"):
                    try:
                        cv_res.update(
                            run_neural_backtest(
                                nixtla_df, horizon, freq, season_length,
                                bt_neural, n_windows, max_steps=30,
                            )
                        )
                        st.success(f"✅ neuralforecast CV done")
                    except Exception as e:
                        st.error(f"neuralforecast CV error: {e}")

            st.session_state["cv_res"] = cv_res

        if st.session_state.get("cv_res"):
            cv_res = st.session_state["cv_res"]

            if not cv_res:
                st.warning("No backtest results. Check for errors above.")
            else:
                # Aggregate leaderboard
                agg_rows = [
                    {"Model": mname, **compute_metrics(cdf["y"].values, cdf["y_pred"].values)}
                    for mname, cdf in cv_res.items()
                ]
                agg_df = pd.DataFrame(agg_rows).sort_values("RMSE").reset_index(drop=True)

                st.subheader("🏆 Model Leaderboard (averaged over all windows)")
                st.dataframe(
                    agg_df.style
                           .format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}%"})
                           .highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#c3e6cb"),
                    use_container_width=True,
                )

                # Bar chart comparison
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(name="MAE",  x=agg_df["Model"], y=agg_df["MAE"],
                                         marker_color="#1f77b4"))
                fig_bar.add_trace(go.Bar(name="RMSE", x=agg_df["Model"], y=agg_df["RMSE"],
                                         marker_color="#ff7f0e"))
                fig_bar.update_layout(
                    barmode="group",
                    title="Model Comparison — MAE & RMSE (backtest avg)",
                    height=350, template="plotly_white",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Per-window breakdown
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
                        st.dataframe(
                            pw_df.style.format(
                                {"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}%"}
                            ),
                            use_container_width=True,
                        )

                # Download
                all_cv = pd.concat(
                    [cdf.assign(Model=m) for m, cdf in cv_res.items()],
                    ignore_index=True,
                )
                st.download_button(
                    "⬇️ Download backtest CSV",
                    all_cv.to_csv(index=False),
                    "backtest_results.csv", "text/csv",
                )


# ══════════════════ TAB: ABOUT ════════════════════
with tab_about:
    st.title("About This App")
    st.markdown("""
## Overview

This app is a **production-style time series forecasting tool** built on
[Nixtla's](https://nixtla.io) open-source ecosystem. Upload any univariate CSV,
select models from three paradigms, and compare their forecast accuracy.

---

## Nixtla Libraries

| Library | Paradigm | Models in this app |
|---------|----------|--------------------|
| **statsforecast** | Classical statistics | AutoARIMA, AutoETS, SeasonalNaive |
| **mlforecast** | Gradient boosting / tree-based ML | LightGBM, XGBoost, RandomForest |
| **neuralforecast** | Deep learning | LSTM, GRU |

All three libraries share the same long-format data schema:

| Column | Description |
|--------|-------------|
| `unique_id` | Series identifier |
| `ds` | Datetime |
| `y` | Target variable (numeric) |

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
```

---

## Train / Test Split

Data is split **chronologically** — the last `test_pct %` of rows become the test set.
No shuffling is performed because temporal order matters in time series.

---

## Backtesting

Uses each library's native `.cross_validation(df, h, n_windows)` method, which rolls a
training window forward `n_windows` times. Each window:
1. Trains on data up to the cutoff date
2. Forecasts the next `h` steps
3. Computes error against the actual future values

Aggregating across windows gives a robust, out-of-sample performance estimate.

---

## Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(|y − ŷ|) | Average error in original units |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more than MAE |
| **MAPE** | mean(|y − ŷ| / y) × 100 % | Scale-free; easy to communicate |

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

1. Push this repo to GitHub (public)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. New app → connect your repo → set **Main file** to `app.py`
4. Click **Deploy** — done!
""")
