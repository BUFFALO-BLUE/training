"""
asof_features.py

Fixes the leakage identified in the master_store_family_dataset.csv approach:
every history-derived feature (average_demand, CV2, ADI, seasonality
strength/consistency, sales_status, demand_pattern, volume_tier, and all
cluster assignments) was originally computed ONCE using the full 2013-2017
window (1,684 calendar days), then joined identically onto every row
regardless of date. A model trained on early-fold rows could see stats that
encode information from the last days of the dataset (2017).

This module recomputes the same *kinds* of features, but as-of a given
cutoff date, using only sales data with date <= asof_date. Call
build_asof_features(daily, asof_date) once per walk-forward fold (using
that fold's train_end as asof_date) and merge the result onto BOTH the
train and validation rows of that fold -- validation rows must also only
ever see train-window statistics, exactly as they would in production
(you don't have tomorrow's data when forecasting tomorrow).

SEASONALITY: two implementations are provided.
  - seasonality_proxies_asof(): cheap, vectorized, ~seconds for all 1,782
    series combined. NOT the method that built the original master CSV.
  - prophet_seasonality_asof(): faithful port of the original
    seasonality_strength.py (Prophet weekly+yearly decomposition, Hyndman-
    Athanasopoulos variance-based strength). One Prophet model per series,
    fit individually -- ~1-3 seconds per series fit (typical published
    Prophet fit times; not benchmarked in this environment, which has no
    network access to install the `prophet` package). For ~1,700 eligible
    series, that's roughly 30-90 minutes for ONE asof cutoff.

    Because this project's walk-forward folds (params.yaml: n_splits=5,
    horizon=15 days) have cutoffs spaced 15 days apart across 5 folds --
    4 gaps of 15 days each, so a total span of 60 days between the
    earliest and latest fold cutoff, out of a ~1,684-day dataset --
    refitting Prophet separately for all 5 folds is unnecessary: call
    prophet_seasonality_asof() ONCE using the EARLIEST fold's cutoff
    (the most conservative choice -- it can't leak into any later fold
    either, since it knows the least), then reuse that single result
    across all 5 folds via the `precomputed_seasonality` argument to
    build_asof_features(). This cuts total Prophet compute time roughly
    5x (to ~30-90 minutes total instead of ~2.5-7.5 hours), at the cost of
    later folds using seasonality estimated with 60 fewer days of history
    than they technically could have. If you later change params.yaml to
    spread folds across different multi-year eras, this shortcut needs
    revisiting -- each era would need its own Prophet fit again.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Same thresholds as the original seasonality_strength.py
PROPHET_MIN_TRAIN_DAYS = 365   # calendar days of history required, minimum
PROPHET_MIN_NONZERO_DAYS = 30  # days with sales > 0 required, minimum


def _weekly_adi_cv2(daily_asof: pd.DataFrame) -> pd.DataFrame:
    """Weekly ADI (sparsity, in weeks-between-sales) and nonzero-only CV^2 (unitless), asof-window only."""
    daily_asof = daily_asof.copy()
    daily_asof["week"] = daily_asof["date"].dt.to_period("W-SUN")
    weekly = daily_asof.groupby(["store_nbr", "family", "week"], as_index=False)["sales"].sum()
    n_weeks = weekly["week"].nunique()  # count of distinct calendar weeks in the asof window

    def sf_stats(g):
        nz = g.loc[g["sales"] > 0, "sales"]
        n_nz = len(nz)  # count of weeks with nonzero total sales
        adi = n_weeks / n_nz if n_nz > 0 else np.nan  # average weeks between nonzero-sales weeks
        if n_nz > 1:
            mean_nz = nz.mean()
            std_nz = nz.std(ddof=1)
            cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else np.nan  # unitless (squared coefficient of variation)
        else:
            cv2 = np.nan
        return pd.Series({"n_nonzero_weeks_asof": n_nz, "ADI_weekly_asof": adi, "CV2_weekly_nonzero_asof": cv2})

    return weekly.groupby(["store_nbr", "family"]).apply(sf_stats).reset_index()


def seasonality_proxies_asof(daily_asof: pd.DataFrame) -> pd.DataFrame:
    """
    Cheap, vectorized proxy for seasonality strength -- NOT the original
    Prophet-based method, see prophet_seasonality_asof() for that. Returns
    a unitless 0-1 strength score: share of total sales variance explained
    by day-of-week / month-of-year group means. Requires at least 14 days
    (2 calendar weeks) for the weekly score, and at least 60 days spanning
    2+ distinct months for the annual score, to avoid a meaningless ratio
    on near-empty series.
    """
    d = daily_asof.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month

    def strength(g, key):
        total_var = g["sales"].var(ddof=0)
        if pd.isna(total_var) or total_var == 0:
            return np.nan
        group_means = g.groupby(key)["sales"].transform("mean")
        explained_var = ((group_means - g["sales"].mean()) ** 2).mean()
        return min(explained_var / total_var, 1.0)

    def sf(g):
        weekly_s = strength(g, "dow") if len(g) >= 14 else np.nan
        annual_s = strength(g, "month") if g["month"].nunique() >= 2 and len(g) >= 60 else np.nan
        return pd.Series({"weekly_seasonality_strength_asof": weekly_s,
                           "annual_seasonality_strength_asof": annual_s})

    return d.groupby(["store_nbr", "family"]).apply(sf).reset_index()


def _prophet_seasonal_strength(residual: np.ndarray, seasonal_component: np.ndarray) -> float:
    """Verbatim port of seasonal_strength() from seasonality_strength.py -- unitless, 0-1."""
    combined = seasonal_component + residual
    var_combined = np.var(combined)
    if var_combined == 0:
        return 0.0
    var_resid = np.var(residual)
    return max(0.0, 1 - (var_resid / var_combined))


def prophet_seasonality_asof(
    daily_asof: pd.DataFrame,
    min_train_days: int = PROPHET_MIN_TRAIN_DAYS,
    min_nonzero_days: int = PROPHET_MIN_NONZERO_DAYS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Faithful port of seasonality_strength.py's run_sweep()/fit_and_score(),
    restricted to daily_asof['date'] <= whatever cutoff daily_asof was
    already filtered to upstream. One Prophet model fit PER (store_nbr,
    family) series -- expect ~1-3 seconds per fit, ~30-90 minutes total
    for ~1,700 eligible series (see module docstring for the full
    time-cost discussion). Requires the `prophet` package installed;
    not runnable in this sandbox (no network access to install it).

    Returns one row per (store_nbr, family) with weekly_seasonality_strength_asof,
    annual_seasonality_strength_asof (both unitless, 0-1), and n_obs_asof
    (count of daily rows used, i.e. elapsed calendar days in the asof window).
    """
    from prophet import Prophet  # imported here so the rest of the module works without it installed

    combos = daily_asof[["store_nbr", "family"]].drop_duplicates().values.tolist()
    total = len(combos)
    if verbose:
        print(f"[prophet_seasonality_asof] fitting {total} series individually "
              f"(this is the slow, exact path -- see module docstring for expected runtime in minutes)")

    rows = []
    for i, (store_nbr, family) in enumerate(combos, 1):
        subset = daily_asof[(daily_asof["store_nbr"] == store_nbr) & (daily_asof["family"] == family)]
        series = subset[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"}).sort_values("ds").reset_index(drop=True)

        row = {"store_nbr": store_nbr, "family": family,
               "n_obs_asof": len(series),  # count of daily rows = elapsed calendar days in this asof window
               "weekly_seasonality_strength_asof": np.nan,
               "annual_seasonality_strength_asof": np.nan}

        n_nonzero = int((series["y"] > 0).sum())  # count of days with sales > 0
        if len(series) >= min_train_days and n_nonzero >= min_nonzero_days and series["y"].sum() > 0:
            model = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                             daily_seasonality=False, interval_width=0.90)
            try:
                model.fit(series)
                forecast = model.predict(series[["ds"]])
                residual = series["y"].values - forecast["yhat"].values
                row["weekly_seasonality_strength_asof"] = round(
                    _prophet_seasonal_strength(residual, forecast["weekly"].values), 4)
                row["annual_seasonality_strength_asof"] = round(
                    _prophet_seasonal_strength(residual, forecast["yearly"].values), 4)
            except Exception:
                pass  # leave NaN, same behavior as the original script's error handling

        rows.append(row)
        if verbose and (i % 100 == 0 or i == total):
            print(f"  [prophet_seasonality_asof] {i}/{total} series fit")

    return pd.DataFrame(rows)


def build_asof_features(
    daily: pd.DataFrame,
    asof_date: pd.Timestamp,
    precomputed_seasonality: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Returns one row per (store_nbr, family) with every history-derived
    feature computed using ONLY daily['date'] <= asof_date.

    precomputed_seasonality: optional DataFrame with columns [store_nbr,
    family, weekly_seasonality_strength_asof, annual_seasonality_strength_asof],
    e.g. the output of prophet_seasonality_asof() computed once at the
    earliest fold's asof_date and reused across folds (see module
    docstring). If None, falls back to the cheap seasonality_proxies_asof().
    """
    daily_asof = daily.loc[daily["date"] <= asof_date, ["store_nbr", "family", "date", "sales"]]

    base = daily_asof.groupby(["store_nbr", "family"]).agg(
        average_demand_asof=("sales", "mean"),
        std_demand_asof=("sales", lambda s: s.std(ddof=1)),
        n_obs_asof=("sales", "size"),  # count of daily rows = elapsed calendar days in this asof window
        zero_days_asof=("sales", lambda s: (s == 0).sum()),  # count of days with sales == 0
    ).reset_index()
    base["n_nonzero_days_asof"] = base["n_obs_asof"] - base["zero_days_asof"]  # count of days
    base["pct_zero_days_asof"] = base["zero_days_asof"] / base["n_obs_asof"]  # unitless fraction, 0-1
    base["cv_asof"] = base["std_demand_asof"] / base["average_demand_asof"].replace(0, np.nan)  # unitless

    wk = _weekly_adi_cv2(daily_asof)

    if precomputed_seasonality is not None:
        ssn = precomputed_seasonality[["store_nbr", "family",
                                        "weekly_seasonality_strength_asof",
                                        "annual_seasonality_strength_asof"]]
    else:
        ssn = seasonality_proxies_asof(daily_asof)

    feat = base.merge(wk, on=["store_nbr", "family"], how="left") \
                .merge(ssn, on=["store_nbr", "family"], how="left")

    # sales_status_asof: identical thresholds to the original definition
    # (0 days / 1-29 days / 30-364 days / 365+ days of nonzero sales),
    # applied to n_nonzero_days_asof computed only through asof_date
    def bucket(n):
        if n == 0:
            return "NO_SALES"
        elif n < 30:
            return "VERY_LOW_SALES"
        elif n < 365:
            return "LOW_SALES"
        else:
            return "SUFFICIENT_SALES"
    feat["sales_status_asof"] = feat["n_nonzero_days_asof"].apply(bucket)

    # demand_pattern_asof: same Syntetos-Boylan cutoffs (ADI < 1.32 weeks, CV2 < 0.49), asof ADI/CV2
    def quadrant(row):
        if pd.isna(row["ADI_weekly_asof"]):
            return "NO_SALES"
        smooth_adi = row["ADI_weekly_asof"] < 1.32
        smooth_cv2 = pd.isna(row["CV2_weekly_nonzero_asof"]) or row["CV2_weekly_nonzero_asof"] < 0.49
        if smooth_adi and smooth_cv2:
            return "SMOOTH"
        elif smooth_adi:
            return "ERRATIC"
        elif smooth_cv2:
            return "INTERMITTENT"
        else:
            return "LUMPY"
    feat["demand_pattern_asof"] = feat.apply(quadrant, axis=1)

    # volume_tier_asof: Pareto ABC (80% / 95% cumulative volume) computed WITHIN this asof snapshot only
    feat["total_volume_asof"] = feat["average_demand_asof"] * feat["n_obs_asof"]  # total units sold, asof window
    ranked = feat.sort_values("total_volume_asof", ascending=False).copy()
    ranked["cum_pct"] = ranked["total_volume_asof"].cumsum() / ranked["total_volume_asof"].sum()  # unitless, 0-1
    def abc(row):
        if row["total_volume_asof"] == 0:
            return "C"
        return "A" if row["cum_pct"] <= 0.80 else ("B" if row["cum_pct"] <= 0.95 else "C")
    ranked["volume_tier_asof"] = ranked.apply(abc, axis=1)
    feat = feat.merge(ranked[["store_nbr", "family", "volume_tier_asof"]], on=["store_nbr", "family"], how="left")

    # hier_cluster_asof: refit k=4 Ward clustering on THIS asof snapshot's sufficient-sales subset only
    cluster_cols = ["average_demand_asof", "CV2_weekly_nonzero_asof", "ADI_weekly_asof",
                     "weekly_seasonality_strength_asof", "annual_seasonality_strength_asof"]
    eligible = feat[(feat["sales_status_asof"] == "SUFFICIENT_SALES")].dropna(subset=cluster_cols).copy()

    feat["hier_cluster_asof"] = feat["sales_status_asof"].map(
        lambda s: "NO_SALES" if s == "NO_SALES" else "INSUFFICIENT_DATA"
    )
    if len(eligible) >= 8:  # need at least 8 rows for a meaningful k=4 fit
        X = eligible[cluster_cols].copy()
        X["average_demand_asof"] = np.log1p(X["average_demand_asof"])
        X_scaled = StandardScaler().fit_transform(X)
        k = min(4, len(eligible) - 1)
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
        eligible["hier_cluster_asof"] = labels.astype(str)
        feat.loc[eligible.index, "hier_cluster_asof"] = eligible["hier_cluster_asof"]

    feat["asof_date"] = asof_date
    return feat.drop(columns=["n_obs_asof", "zero_days_asof", "total_volume_asof", "cum_pct"], errors="ignore")


def attach_asof_features(rows: pd.DataFrame, daily: pd.DataFrame, asof_date: pd.Timestamp,
                          precomputed_seasonality: pd.DataFrame = None) -> pd.DataFrame:
    """Compute asof features once, join onto any subset of rows (train or val) for that fold."""
    feat = build_asof_features(daily, asof_date, precomputed_seasonality=precomputed_seasonality)
    return rows.merge(feat, on=["store_nbr", "family"], how="left")