"""
train_prophet.py

Fits one Prophet model PER SERIES (store_nbr x family), same per-series
approach as ARIMA -- 1,782 series x n_folds = thousands of individual
Prophet fits (e.g. 1,782 x 5 folds = 8,910 fits).

Prophet requires columns named exactly `ds` (date) and `y` (target) --
handled internally, doesn't change your master training table.

Regressors: is_holiday and onpromotion are added as extra regressors
since Prophet supports this natively and you already have both aligned
to the daily grid.

Tracking: writes metrics_prophet.json / fold_metrics_prophet.csv,
namespaced separately from LightGBM/ARIMA outputs.

Run: python train_prophet.py
NOTE: at this series count, this is the slowest of the three scripts to
run end to end -- start with a small n_jobs test before scaling up, and
budget real wall-clock time (thousands of independent model fits).
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from prophet import Prophet

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

from build_training_set import load_training_table, walk_forward_splits

TARGET = "sales"


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def asymmetric_cost(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    residual = y_true - y_pred
    loss = np.where(residual > 0, alpha * residual**2, residual**2)
    return float(np.mean(loss))


def fit_and_score_one_series(
    series_df: pd.DataFrame,
    train_dates: list,
    val_dates: list,
    alpha: float,
    prophet_params: dict,
) -> dict:
    store_nbr = series_df["store_nbr"].iloc[0]
    family = series_df["family"].iloc[0]

    train = series_df[series_df["date"].isin(train_dates)].sort_values("date")
    val = series_df[series_df["date"].isin(val_dates)].sort_values("date")

    if len(train) < prophet_params["min_train_days"] or len(val) == 0:
        return None

    train_prophet = train.rename(columns={"date": "ds", TARGET: "y"})[
        ["ds", "y", "is_holiday", "onpromotion"]
    ]
    val_prophet = val.rename(columns={"date": "ds", TARGET: "y"})[
        ["ds", "y", "is_holiday", "onpromotion"]
    ]

    try:
        model = Prophet(
            weekly_seasonality=prophet_params["weekly_seasonality"],
            yearly_seasonality=prophet_params["yearly_seasonality"],
            daily_seasonality=False,
            changepoint_prior_scale=prophet_params["changepoint_prior_scale"],
        )
        model.add_regressor("is_holiday")
        model.add_regressor("onpromotion")
        model.fit(train_prophet)

        future = val_prophet[["ds", "is_holiday", "onpromotion"]]
        forecast = model.predict(future)
        y_pred = np.clip(forecast["yhat"].values, a_min=0, a_max=None)
    except Exception as e:
        return {
            "store_nbr": store_nbr, "family": family,
            "status": "failed", "error": str(e), "asymmetric_cost": None,
        }

    cost = asymmetric_cost(val_prophet["y"].values, y_pred, alpha)

    return {
        "store_nbr": store_nbr, "family": family,
        "status": "ok", "error": None, "asymmetric_cost": cost,
    }


def train_all_folds_prophet(params: dict):
    alpha = params["asymmetric_loss"]["alpha"]
    n_splits = params["walk_forward"]["n_splits"]
    horizon = params["walk_forward"]["horizon"]
    gap = params["walk_forward"]["gap"]
    prophet_params = params["prophet"]

    print("Loading merged training table...")
    table = load_training_table()
    table["is_holiday"] = table["is_holiday"].astype(int)  # Prophet regressors must be numeric

    splits = walk_forward_splits(table["date"], n_splits=n_splits, horizon=horizon, gap=gap)
    series_groups = list(table.groupby(["store_nbr", "family"], sort=False))
    total_fits = len(series_groups) * len(splits)
    print(f"{len(series_groups)} series x {len(splits)} folds = {total_fits} total Prophet fits")

    fold_records = []

    for fold_idx, (train_dates, val_dates) in enumerate(splits):
        print(f"Fold {fold_idx}: fitting {len(series_groups)} Prophet models in parallel...")

        results = Parallel(n_jobs=prophet_params["n_jobs"], verbose=5)(
            delayed(fit_and_score_one_series)(
                series_df[["date", "store_nbr", "family", TARGET, "is_holiday", "onpromotion"]],
                train_dates, val_dates, alpha, prophet_params,
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

        pd.DataFrame(results).to_csv(f"prophet_series_detail_fold{fold_idx}.csv", index=False)

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("fold_metrics_prophet.csv", index=False)

    summary = {
        "model_type": "prophet",
        "mean_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].mean()),
        "std_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].std()),
        "alpha": alpha,
        "n_folds": len(fold_records),
    }
    with open("metrics_prophet.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean asymmetric MSE across folds (Prophet): {summary['mean_val_asymmetric_mse']:.4f}")
    return fold_records


if __name__ == "__main__":
    params = load_params()
    train_all_folds_prophet(params)