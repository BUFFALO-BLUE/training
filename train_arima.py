"""
train_arima.py

Fits one ARIMA model PER SERIES (store_nbr x family) -- unlike the global
LightGBM model, ARIMA is not a panel model, so this is 1,782 independent
fits. Uses auto_arima for order selection (manual (p,d,q) tuning across
1,782 series is not realistic).

Evaluated on the same walk-forward folds and the same asymmetric cost
metric as LightGBM, so `dvc metrics show` gives an apples-to-apples
comparison between model types.

Tracking: writes metrics_arima.json / fold_metrics_arima.csv, namespaced
separately from the LightGBM outputs so both show up side by side under
`dvc metrics show` / `dvc plots show`.

Run: python train_arima.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
import yaml
from joblib import Parallel, delayed

from build_training_set import load_training_table, walk_forward_splits

warnings.filterwarnings("ignore")  # auto_arima is noisy about convergence warnings on sparse series

TARGET = "sales"


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def asymmetric_cost(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Same asymmetric MSE used for LightGBM -- required for a fair model_type comparison."""
    residual = y_true - y_pred
    loss = np.where(residual > 0, alpha * residual**2, residual**2)
    return float(np.mean(loss))


def fit_and_score_one_series(
    series_df: pd.DataFrame,
    train_dates: list,
    val_dates: list,
    alpha: float,
    arima_params: dict,
) -> dict:
    """Fit auto_arima on one (store, family) series for one fold, score on validation horizon."""
    store_nbr = series_df["store_nbr"].iloc[0]
    family = series_df["family"].iloc[0]

    train = series_df[series_df["date"].isin(train_dates)].sort_values("date")
    val = series_df[series_df["date"].isin(val_dates)].sort_values("date")

    if len(train) < arima_params["min_train_days"] or len(val) == 0:
        return None  # not enough history yet for this series/fold -- skip, don't fabricate a score

    y_train = train[TARGET].values
    y_val = val[TARGET].values

    try:
        model = pm.auto_arima(
            y_train,
            seasonal=arima_params["seasonal"],
            m=arima_params["seasonal_period"],
            max_p=arima_params["max_p"],
            max_q=arima_params["max_q"],
            max_d=arima_params["max_d"],
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
        )
        y_pred = model.predict(n_periods=len(y_val))
        y_pred = np.clip(y_pred, a_min=0, a_max=None)  # sales can't be negative
    except Exception as e:
        return {
            "store_nbr": store_nbr, "family": family,
            "status": "failed", "error": str(e),
            "asymmetric_cost": None, "order": None,
        }

    cost = asymmetric_cost(y_val, y_pred, alpha)

    return {
        "store_nbr": store_nbr, "family": family,
        "status": "ok", "error": None,
        "asymmetric_cost": cost, "order": str(model.order),
    }


def train_all_folds_arima(params: dict):
    alpha = params["asymmetric_loss"]["alpha"]
    n_splits = params["walk_forward"]["n_splits"]
    horizon = params["walk_forward"]["horizon"]
    gap = params["walk_forward"]["gap"]
    arima_params = params["arima"]

    print("Loading merged training table...")
    table = load_training_table()
    splits = walk_forward_splits(table["date"], n_splits=n_splits, horizon=horizon, gap=gap)

    series_groups = list(table.groupby(["store_nbr", "family"], sort=False))
    print(f"{len(series_groups)} series to fit per fold, {len(splits)} folds, "
          f"{len(series_groups) * len(splits)} total ARIMA fits")

    fold_records = []

    for fold_idx, (train_dates, val_dates) in enumerate(splits):
        print(f"Fold {fold_idx}: fitting {len(series_groups)} series in parallel...")

        results = Parallel(n_jobs=arima_params["n_jobs"], verbose=5)(
            delayed(fit_and_score_one_series)(
                series_df[["date", "store_nbr", "family", TARGET]],
                train_dates, val_dates, alpha, arima_params,
            )
            for _, series_df in series_groups
        )
        results = [r for r in results if r is not None]

        ok_results = [r for r in results if r["status"] == "ok"]
        failed_count = len(results) - len(ok_results)
        mean_cost = float(np.mean([r["asymmetric_cost"] for r in ok_results])) if ok_results else None

        fold_records.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0]), "train_end": str(train_dates[-1]),
            "val_start": str(val_dates[0]), "val_end": str(val_dates[-1]),
            "n_series_fit": len(ok_results),
            "n_series_failed": failed_count,
            "val_asymmetric_mse": mean_cost,
        })
        print(f"Fold {fold_idx}: mean asymmetric cost = {mean_cost}, "
              f"{len(ok_results)} ok, {failed_count} failed")

        # per-series detail, one file per fold -- useful for spotting which
        # clusters/series ARIMA struggles on vs. LightGBM
        pd.DataFrame(results).to_csv(f"arima_series_detail_fold{fold_idx}.csv", index=False)

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("fold_metrics_arima.csv", index=False)

    summary = {
        "model_type": "arima",
        "mean_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].mean()),
        "std_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].std()),
        "alpha": alpha,
        "n_folds": len(fold_records),
    }
    with open("metrics_arima.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean asymmetric MSE across folds (ARIMA): {summary['mean_val_asymmetric_mse']:.4f}")
    return fold_records


if __name__ == "__main__":
    params = load_params()
    train_all_folds_arima(params)