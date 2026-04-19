import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred):
    a = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    e = a - p
    mae  = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mape = float(np.mean(np.abs(e / a)) * 100) if np.all(a != 0) else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def make_leaderboard(results_dict):
    rows = [{"Model": name, **m} for name, m in results_dict.items()]
    df = pd.DataFrame(rows)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE").reset_index(drop=True)
    return df
