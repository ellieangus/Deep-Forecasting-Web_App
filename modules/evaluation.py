import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred, y_train=None):
    """Compute MAE, RMSE, MAPE, sMAPE, and (if y_train provided) MASE for a forecast."""
    a = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    nan5 = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"),
             "sMAPE": float("nan"), "MASE": float("nan")}
    if len(a) == 0:
        return nan5
    e = a - p
    mae  = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mape = float(np.mean(np.abs(e / a)) * 100) if np.all(a != 0) else float("nan")
    denom = np.abs(a) + np.abs(p)
    smape = float(np.mean(200 * np.abs(e) / denom)) if np.all(denom != 0) else float("nan")
    mase = float("nan")
    if y_train is not None:
        tr = np.array(y_train, dtype=float)
        naive_mae = np.mean(np.abs(np.diff(tr)))
        if naive_mae > 0:
            mase = float(mae / naive_mae)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "MASE": mase}


def make_leaderboard(results_dict):
    """Build a DataFrame leaderboard sorted by RMSE from a {model_name: metrics} dict."""
    rows = [{"Model": name, **m} for name, m in results_dict.items()]
    df = pd.DataFrame(rows)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE").reset_index(drop=True)
    return df
