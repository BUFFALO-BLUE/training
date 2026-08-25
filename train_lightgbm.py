"""
train_lightgbm.py

Trains the global LightGBM model across all walk-forward folds using
the asymmetric (underprediction-penalized) custom objective.

Tracking is DVC-native (not MLflow):
  - Hyperparameters live in params.yaml (dvc reads this automatically)
  - Per-fold + overall metrics are written to metrics.json / fold_metrics.csv
    (both are declared as `metrics`/`plots` outputs in dvc.yaml)
  - Model boosters are saved under models/ (declared as a `outs` in dvc.yaml)

Run directly for local iteration: python train_lightgbm.py
Run as part of the pipeline: dvc repro   (or dvc exp run for experiment tracking)
"""

import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

from build_training_set import load_training_table, walk_forward_splits

TARGET = "sales"

NON_FEATURE_COLS = [
    "id", "date", TARGET,
    "first_nonzero_date", "last_nonzero_date",
]


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
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def get_categorical_columns(df: pd.DataFrame, feature_cols: list) -> list:
    return [c for c in feature_cols if str(df[c].dtype) == "category"]


def train_all_folds(params: dict):
    alpha = params["asymmetric_loss"]["alpha"]
    n_splits = params["walk_forward"]["n_splits"]
    horizon = params["walk_forward"]["horizon"]
    gap = params["walk_forward"]["gap"]
    lgb_params = params["lightgbm"]

    print("Loading merged training table...")
    table = load_training_table()

    feature_cols = get_feature_columns(table)
    cat_cols = get_categorical_columns(table, feature_cols)
    print(f"{len(feature_cols)} features, {len(cat_cols)} categorical")

    splits = walk_forward_splits(table["date"], n_splits=n_splits, horizon=horizon, gap=gap)

    fold_records = []

    for fold_idx, (train_dates, val_dates) in enumerate(splits):
        train_mask = table["date"].isin(train_dates)
        val_mask = table["date"].isin(val_dates)

        X_train, y_train = table.loc[train_mask, feature_cols], table.loc[train_mask, TARGET]
        X_val, y_val = table.loc[val_mask, feature_cols], table.loc[val_mask, TARGET]

        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

        # Newer LightGBM (4.x+) removed the standalone `fobj` argument from
        # train(). A custom objective is now set as params['objective']
        # itself (a callable), the same slot that would otherwise hold a
        # string like 'regression'. feval is unaffected and still passed
        # separately.
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
            "best_iteration": booster.best_iteration,
            "val_asymmetric_mse": val_score,
        })
        print(f"Fold {fold_idx}: val_asymmetric_mse = {val_score:.4f} (best_iter={booster.best_iteration})")

    # Per-fold metrics -> CSV, tracked as a `dvc plots` output (see dvc.yaml)
    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("fold_metrics.csv", index=False)

    # Summary metrics -> JSON, tracked as a `dvc metrics` output
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