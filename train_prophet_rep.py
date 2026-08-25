"""
train_prophet_representative.py

Fits Prophet on a small, representative subset of series -- ONE (or a few)
per cluster, picked as the series closest to that cluster's centroid in
feature space (a medoid) -- rather than all 1,782 series. This is a
sanity-check comparison, not the full-scale Prophet run: does a
cluster-specialized model meaningfully beat the pooled global LightGBM on
series that are "typical" of their cluster?

Selection uses the ORIGINAL (non-asof) cluster columns from the master
file -- picking which series to study is a one-time design decision, not
a model input, so it isn't subject to the walk-forward leakage concern
that applies to features the model actually trains on.

Run: python train_prophet_representative.py
"""

import json
import warnings

import numpy as np
import pandas as pd
import yaml
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

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


def select_representative_series(
    table: pd.DataFrame, cluster_col: str, n_per_cluster: int, selection_features: list
) -> pd.DataFrame:
    """
    Picks the n_per_cluster series closest (Euclidean, standardized) to
    each cluster's centroid -- the "typical" series for that cluster,
    not the biggest or the first alphabetically. Excludes sentinel
    non-cluster values (NO_SALES, INSUFFICIENT_DATA, NOT_APPLICABLE) --
    there's no meaningful centroid for those groups.
    """
    series_level = table[["store_nbr", "family", cluster_col] + selection_features].drop_duplicates(
        subset=["store_nbr", "family"]
    )
    sentinels = {"NO_SALES", "INSUFFICIENT_DATA", "NOT_APPLICABLE"}
    series_level = series_level[~series_level[cluster_col].astype(str).isin(sentinels)]
    series_level = series_level.dropna(subset=selection_features)

    picks = []
    for cluster_id, group in series_level.groupby(cluster_col):
        if len(group) == 0:
            continue
        X = group[selection_features].copy()
        if "average_demand" in X.columns:
            X["average_demand"] = np.log1p(X["average_demand"])
        X_scaled = StandardScaler().fit_transform(X)
        centroid = X_scaled.mean(axis=0)
        dist = np.linalg.norm(X_scaled - centroid, axis=1)
        group = group.assign(_dist_to_centroid=dist).sort_values("_dist_to_centroid")
        picks.append(group.head(n_per_cluster))

    return pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=series_level.columns)


def fit_and_score_one_series(
    series_df: pd.DataFrame, train_dates: list, val_dates: list, alpha: float, prophet_params: dict,
) -> dict:
    train = series_df[series_df["date"].isin(train_dates)].sort_values("date")
    val = series_df[series_df["date"].isin(val_dates)].sort_values("date")

    if len(train) < prophet_params["min_train_days"] or len(val) == 0:
        return None

    train_p = train.rename(columns={"date": "ds", TARGET: "y"})[["ds", "y", "is_holiday", "onpromotion"]]
    val_p = val.rename(columns={"date": "ds", TARGET: "y"})[["ds", "y", "is_holiday", "onpromotion"]]

    try:
        model = Prophet(
            weekly_seasonality=prophet_params["weekly_seasonality"],
            yearly_seasonality=prophet_params["yearly_seasonality"],
            daily_seasonality=False,
            changepoint_prior_scale=prophet_params["changepoint_prior_scale"],
        )
        model.add_regressor("is_holiday")
        model.add_regressor("onpromotion")
        model.fit(train_p)
        forecast = model.predict(val_p[["ds", "is_holiday", "onpromotion"]])
        y_pred = np.clip(forecast["yhat"].values, a_min=0, a_max=None)
    except Exception as e:
        return {"status": "failed", "error": str(e), "asymmetric_cost": None}

    return {"status": "ok", "error": None, "asymmetric_cost": asymmetric_cost(val_p["y"].values, y_pred, alpha)}


def train_all_folds_prophet_representative(params: dict):
    alpha = params["asymmetric_loss"]["alpha"]
    n_splits = params["walk_forward"]["n_splits"]
    horizon = params["walk_forward"]["horizon"]
    gap = params["walk_forward"]["gap"]
    rep_params = params["prophet_representative"]

    print("Loading merged training table...")
    table = load_training_table()
    table["is_holiday"] = table["is_holiday"].astype(int)

    representatives = select_representative_series(
        table,
        cluster_col=rep_params["cluster_column"],
        n_per_cluster=rep_params["n_per_cluster"],
        selection_features=rep_params["selection_features"],
    )
    print(f"Selected {len(representatives)} representative series across "
          f"{representatives[rep_params['cluster_column']].nunique()} clusters "
          f"(cluster_column={rep_params['cluster_column']}):")
    print(representatives[["store_nbr", "family", rep_params["cluster_column"]]].to_string(index=False))

    splits = walk_forward_splits(table["date"], n_splits=n_splits, horizon=horizon, gap=gap)

    fold_records = []
    for fold_idx, (train_dates, val_dates) in enumerate(splits):
        print(f"Fold {fold_idx}: fitting {len(representatives)} representative Prophet models...")
        per_series_results = []

        for _, rep in representatives.iterrows():
            series_df = table[(table["store_nbr"] == rep["store_nbr"]) & (table["family"] == rep["family"])][
                ["date", TARGET, "is_holiday", "onpromotion"]
            ]
            result = fit_and_score_one_series(series_df, train_dates, val_dates, alpha, rep_params)
            if result is None:
                continue
            result.update({
                "store_nbr": rep["store_nbr"], "family": rep["family"],
                "cluster": rep[rep_params["cluster_column"]],
            })
            per_series_results.append(result)

        ok_results = [r for r in per_series_results if r["status"] == "ok"]
        mean_cost = float(np.mean([r["asymmetric_cost"] for r in ok_results])) if ok_results else None

        fold_records.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0]), "train_end": str(train_dates[-1]),
            "val_start": str(val_dates[0]), "val_end": str(val_dates[-1]),
            "n_series_fit": len(ok_results),
            "val_asymmetric_mse": mean_cost,
        })
        print(f"Fold {fold_idx}: mean asymmetric cost = {mean_cost}")

        pd.DataFrame(per_series_results).to_csv(f"prophet_representative_detail_fold{fold_idx}.csv", index=False)

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("fold_metrics_prophet_representative.csv", index=False)

    summary = {
        "model_type": "prophet_representative",
        "cluster_column": rep_params["cluster_column"],
        "n_representative_series": len(representatives),
        "mean_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].mean()),
        "std_val_asymmetric_mse": float(fold_df["val_asymmetric_mse"].std()),
        "alpha": alpha,
        "n_folds": len(fold_records),
    }
    with open("metrics_prophet_representative.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean asymmetric MSE across folds (Prophet, representative series): "
          f"{summary['mean_val_asymmetric_mse']:.4f}")
    return fold_records


if __name__ == "__main__":
    params = load_params()
    train_all_folds_prophet_representative(params)