"""
train_sarima_by_cluster.py

Case study: does forecasting accuracy improve when SARIMA is fit at the
CLUSTER level rather than the individual series level?

Unlike train_arima.py (one SARIMA per store x family, cluster membership
unused), this script aggregates daily sales UP to the cluster level first
(summing all series in a cluster), then fits one auto-tuned SARIMA per
cluster per fold. Far fewer, larger, more stable series -- a genuinely
different modeling granularity, not just a rename of the per-series script.

auto_arima with seasonal=True already performs the "auto-tuning" (a
stepwise search over (p,d,q)(P,D,Q,m) by AIC/BIC) -- no separate tuning
step needed.

Exogenous regressors (oil_price, is_holiday, total onpromotion count) are
aggregated per cluster per day and passed to auto_arima via `X`, making
this technically SARIMAX -- still commonly just called "SARIMA with
exogenous regressors" in practice.

Run: python train_sarima_by_cluster.py
"""

import json
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
import yaml
from joblib import Parallel, delayed

from build_training_set import load_training_table, walk_forward_splits

warnings.filterwarnings("ignore")

TARGET = "sales"



def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def asymmetric_cost(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    residual = y_true - y_pred
    loss = np.where(residual > 0, alpha * residual**2, residual**2)
    return float(np.mean(loss))


def aggregate_by_cluster(table: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Collapses the 1,782-series daily panel up to one row per (cluster, date),
    summing sales/onpromotion and averaging is_holiday/oil_price (both are
    already identical across series on a given date, so mean == the value).
    """
    agg = (
        table.groupby([cluster_col, "date"])
        .agg(
            sales=(TARGET, "sum"),
            onpromotion=("onpromotion", "sum"),
            is_holiday=("is_holiday", "mean"),
            oil_price=("oil_price", "mean"),
        )
        .reset_index()
    )
    agg["is_holiday"] = agg["is_holiday"].round().astype(int)
    return agg


def fit_and_score_one_cluster(
    cluster_df: pd.DataFrame,
    cluster_id: str,
    train_dates: list,
    val_dates: list,
    alpha: float,
    sarima_params: dict,
) -> dict:
    train = cluster_df[cluster_df["date"].isin(train_dates)].sort_values("date")
    val = cluster_df[cluster_df["date"].isin(val_dates)].sort_values("date")

    if len(train) < sarima_params["min_train_days"] or len(val) == 0:
        return None

    y_train = train["sales"].values
    y_val = val["sales"].values

    exog_cols = ["onpromotion", "is_holiday", "oil_price"]
    X_train = train[exog_cols].values if sarima_params["use_exogenous"] else None
    X_val = val[exog_cols].values if sarima_params["use_exogenous"] else None

    try:
        model = pm.auto_arima(
            y_train,
            X=X_train,
            seasonal=sarima_params["seasonal"],
            m=sarima_params["seasonal_period"],
            max_p=sarima_params["max_p"],
            max_q=sarima_params["max_q"],
            max_d=sarima_params["max_d"],
            max_P=sarima_params["max_P"],
            max_Q=sarima_params["max_Q"],
            max_D=sarima_params["max_D"],
            information_criterion=sarima_params["information_criterion"],
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
        )
        y_pred = model.predict(n_periods=len(y_val), X=X_val)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
    except Exception as e:
        return {
            "cluster": cluster_id, "status": "failed", "error": str(e),
            "asymmetric_cost": None, "order": None, "seasonal_order": None,
        }

    cost = asymmetric_cost(y_val, y_pred, alpha)

    return {
        "cluster": cluster_id, "status": "ok", "error": None,
        "asymmetric_cost": cost,
        "order": str(model.order),
        "seasonal_order": str(model.seasonal_order),
    }


def train_all_folds_sarima_cluster(params: dict):
    alpha = params["asymmetric_loss"]["alpha"]
    n_splits = params["walk_forward"]["n_splits"]
    horizon = params["walk_forward"]["horizon"]
    gap = params["walk_forward"]["gap"]
    sarima_params = params["sarima_cluster"]
    cluster_col = sarima_params["cluster_column"]

    print("Loading merged training table...")
    table = load_training_table()

    print(f"Aggregating to cluster level using '{cluster_col}'...")
    cluster_table = aggregate_by_cluster(table, cluster_col)
    clusters = sorted(cluster_table[cluster_col].unique())
    print(f"{len(clusters)} clusters found: {clusters}")

    splits = walk_forward_splits(cluster_table["date"], n_splits=n_splits, horizon=horizon, gap=gap)
    print(f"{len(clusters)} clusters x {len(splits)} folds = {len(clusters) * len(splits)} total SARIMA fits")

    fold_records = []

    for fold_idx, (train_dates, val_dates) in enumerate(splits):
        print(f"Fold {fold_idx}: fitting {len(clusters)} cluster-level SARIMA models in parallel...")

        results = Parallel(n_jobs=sarima_params["n_jobs"], verbose=5)(
            delayed(fit_and_score_one_cluster)(
                cluster_table[cluster_table[cluster_col] == c],
                c, train_dates, val_dates, alpha, sarima_params,
            )
            for c in clusters
        )
        results = [r for r in results if r is not None]

        ok_results = [r for r in results if r["status"] == "ok"]
        failed_count = len(results) - len(ok_results)
        mean_cost = float(np.mean([r["asymmetric_cost"] for r in ok_results])) if ok_results else None

        fold_records.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0]), "train_end": str(train_dates[-1]),
            "val_start": str(val_dates[0]), "val_end": str(val_dates[-1]),
            "n_clusters_fit": len(ok_results),
            "n_clusters_failed": failed_count,
            "val_asymmetric_mse": mean_cost,
        })
        print(f"Fold {fold_idx}: mean asymmetric cost = {mean_cost}, "
              f"{len(ok_results)} ok, {failed_count} failed")

        # per-cluster detail -- this IS the case study: which clusters does
        # SARIMA handle well vs. poorly, and what orders did auto-tuning pick
        pd.DataFrame(results).to_csv(f"sarima_cluster_detail_fold{fold_idx}.csv", index=False)

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("fold_metrics_sarima_cluster.csv", index=False)

    summary = {
        "model_type": "sarima_by_cluster",
        "cluster_column": cluster_col,
        "n_clusters": len(clusters),
        "mean_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].mean()),
        "std_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].std()),
        "alpha": alpha,
        "n_folds": len(fold_records),
    }
    with open("metrics_sarima_cluster.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean asymmetric MSE across folds (SARIMA by {cluster_col}): "
          f"{summary['mean_val_asymmetric_mse']:.4f}")
    return fold_records





if __name__ == "__main__":
    params = load_params()
    train_all_folds_sarima_cluster(params)    
