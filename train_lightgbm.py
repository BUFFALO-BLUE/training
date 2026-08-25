"""
train_lightgbm.py  (leakage-fixed)

Change from the original version: the master CSV's history-derived columns
(average_demand, CV2, ADI, seasonality strength/consistency, sales_status,
demand_pattern, volume_tier, and all cluster assignments) were computed ONCE
over the full 2013-2017 window and joined identically onto every row. That
leaks future information into every walk-forward fold, even early ones.

Fix: those columns are excluded from the loaded table (LEAKY_STATIC_COLS)
and instead recomputed per fold, using only that fold's training window,
via asof_features.build_asof_features(). Both train AND val rows in a fold
get the SAME asof-computed features (asof_date = that fold's train_end) --
validation rows must only ever see train-window statistics, exactly like
production, where you don't have tomorrow's data when forecasting tomorrow.
"""

import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

from build_training_set import load_training_table, walk_forward_splits
from asof_features import build_asof_features, prophet_seasonality_asof

TARGET = "sales"

NON_FEATURE_COLS = ["id", "date", TARGET]

# Computed once over the FULL window in the master CSV -> leaks the future.
# Excluded here; fold-specific *_asof replacements are attached in the loop.
LEAKY_STATIC_COLS = [
    "average_demand", "std_demand", "coefficient_of_variation",
    "pct_zero_days", "zero_sales_days", "number_of_observations",
    "days_of_history", "pct_of_window_covered", "pct_zero_within_active_window",
    "n_nonzero_weeks", "ADI_weekly", "CV2_weekly_nonzero",
    "weekly_seasonality_strength", "annual_seasonality_strength",
    "annual_seasonality_consistency", "n_full_years_used",
    "weekly_seasonality_consistency", "n_years_used_weekly",
    "sales_status", "demand_pattern", "volume_tier",
    "hier_cluster", "volume_cluster", "nested_cluster", "sparse_cluster",
]

ASOF_CATEGORICAL_COLS = ["sales_status_asof", "demand_pattern_asof", "volume_tier_asof", "hier_cluster_asof"]


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def asymmetric_mse_objective(y_pred, dataset, alpha: float):
    y_true = dataset.get_label()
    residual = y_true - y_pred
    grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
    hess = np.where(residual > 0, 2 * alpha, 2.0)
    return grad, hess


def asymmetric_mse_eval(y_pred, dataset, alpha: float):
    y_true = dataset.get_label()
    residual = y_true - y_pred
    loss = np.where(residual > 0, alpha * residual**2, residual**2)
    return "asymmetric_mse", float(np.mean(loss)), False


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and c not in LEAKY_STATIC_COLS]


def get_categorical_columns(df: pd.DataFrame, feature_cols: list) -> list:
    return [c for c in feature_cols if str(df[c].dtype) in ("category", "object", "bool")]


def attach_fold_asof_features(table: pd.DataFrame, train_mask, val_mask, asof_date: pd.Timestamp,
                               precomputed_seasonality: pd.DataFrame = None):
    """
    Computes asof features once (using only rows with date <= asof_date,
    i.e. this fold's training window) and merges them onto both the train
    and validation slices of this fold. Categorical dtype is set from the
    asof table itself so train/val share an identical category set.
    """
    daily_for_asof = table.loc[table["date"] <= asof_date, ["store_nbr", "family", "date", "sales"]]
    asof_feat = build_asof_features(daily_for_asof, asof_date, precomputed_seasonality=precomputed_seasonality)

    for c in ASOF_CATEGORICAL_COLS:
        asof_feat[c] = asof_feat[c].astype("category")

    train_df = table.loc[train_mask].merge(asof_feat, on=["store_nbr", "family"], how="left")
    val_df = table.loc[val_mask].merge(asof_feat, on=["store_nbr", "family"], how="left")
    return train_df, val_df


def train_all_folds(params: dict):
    alpha = params["asymmetric_loss"]["alpha"]
    n_splits = params["walk_forward"]["n_splits"]
    horizon = params["walk_forward"]["horizon"]
    gap = params["walk_forward"]["gap"]
    lgb_params = params["lightgbm"]

    print("Loading merged training table...")
    table = load_training_table()

    splits = walk_forward_splits(table["date"], n_splits=n_splits, horizon=horizon, gap=gap)

    # Prophet seasonality is expensive (~1-3 sec per series fit, ~1,700
    # eligible series = ~30-90 minutes for ONE cutoff). All 5 folds' train
    # windows end within `horizon * n_splits` = 75 days of each other, so
    # compute it ONCE at the EARLIEST fold's cutoff (most conservative --
    # can't leak into any later fold either) and reuse across all folds.
    earliest_asof_date = pd.Timestamp(splits[0][0][-1])
    print(f"Computing Prophet seasonality once, as of {earliest_asof_date.date()} "
          f"(reused across all {len(splits)} folds -- see asof_features.py docstring)")
    daily_full = table[["store_nbr", "family", "date", "sales"]]
    daily_for_prophet = daily_full.loc[daily_full["date"] <= earliest_asof_date]
    shared_seasonality = prophet_seasonality_asof(daily_for_prophet)

    fold_records = []

    for fold_idx, (train_dates, val_dates) in enumerate(splits):
        train_mask = table["date"].isin(train_dates)
        val_mask = table["date"].isin(val_dates)
        asof_date = pd.Timestamp(train_dates[-1])

        print(f"Fold {fold_idx}: computing asof features as of {asof_date.date()} "
              f"(train window: {train_dates[0]} .. {asof_date.date()})")
        train_df, val_df = attach_fold_asof_features(
            table, train_mask, val_mask, asof_date, precomputed_seasonality=shared_seasonality
        )

        feature_cols = get_feature_columns(train_df)
        cat_cols = get_categorical_columns(train_df, feature_cols)

        X_train, y_train = train_df[feature_cols], train_df[TARGET]
        X_val, y_val = val_df[feature_cols], val_df[TARGET]

        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

        fold_lgb_params = dict(lgb_params)
        fold_lgb_params["objective"] = lambda p, d: asymmetric_mse_objective(p, d, alpha)

        booster = lgb.train(
            fold_lgb_params,
            train_set,
            num_boost_round=params["training"]["num_boost_round"],
            valid_sets=[val_set],
            feval=lambda p, d: asymmetric_mse_eval(p, d, alpha),
            callbacks=[lgb.early_stopping(stopping_rounds=params["training"]["early_stopping_rounds"], verbose=False)],
        )

        val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
        _, val_score, _ = asymmetric_mse_eval(val_pred, val_set, alpha)

        booster.save_model(f"models/lgbm_fold{fold_idx}.txt")

        fold_records.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "val_start": str(val_dates[0]),
            "val_end": str(val_dates[-1]),
            "n_features": len(feature_cols),
            "n_categorical": len(cat_cols),
            "best_iteration": booster.best_iteration,
            "val_asymmetric_mse": val_score,
        })
        print(f"Fold {fold_idx}: val_asymmetric_mse = {val_score:.4f} (best_iter={booster.best_iteration})")

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("fold_metrics.csv", index=False)

    summary = {
        "mean_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].mean()),
        "std_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].std()),
        "alpha": alpha,
        "n_folds": len(fold_records),
    }
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Mean asymmetric MSE across {len(fold_records)} folds: {summary['mean_val_asymmetric_mse']:.4f}")
    return fold_records


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    params = load_params()
    train_all_folds(params)