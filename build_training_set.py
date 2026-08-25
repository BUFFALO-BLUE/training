import os
import re
import time

import numpy as np
import pandas as pd

DAILY_PATH = "/workspaces/training/data/merged_data.csv"
MASTER_PATH = "/workspaces/training/data/master_store_family_dataset.csv"

MASTER_DROP_COLS = [
    "wape_D", "wape_W", "wape_ME",
    "total_volume",
    "n_nonzero_days", "never_sold",
    "city", "state", "type", "cluster",
    # raw text -- not model-usable as-is, parsed into numeric features below
    "price_range_usd", "shelf_life", "representative_item",
    # per-series date strings -- days_of_history / pct_of_window_covered
    # already capture this info numerically; raw date strings would need
    # to be recomputed relative to each row's own `date` to avoid leaking
    # future info, so they're dropped rather than used as-is
    "first_nonzero_date", "last_nonzero_date",
]

STATIC_CATEGORICAL_COLS = [
    "sales_status", "demand_pattern", "volume_tier",
    "hier_cluster", "volume_cluster", "nested_cluster", "sparse_cluster",
    "perishability", "durability",
]

DAILY_CATEGORICAL_COLS = ["store_nbr", "family", "city", "state", "type", "cluster"]

_SHELF_LIFE_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7,
                          "month": 30, "months": 30, "year": 365, "years": 365, "yrs": 365}


def parse_price_range(master: pd.DataFrame) -> pd.DataFrame:
    """price_range_usd like '4-10' or '0.80-2.00' -> price_min, price_max (float)."""
    parts = master["price_range_usd"].str.split("-", n=1, expand=True)
    master["price_min"] = pd.to_numeric(parts[0], errors="coerce")
    master["price_max"] = pd.to_numeric(parts[1], errors="coerce")
    return master


def parse_shelf_life(master: pd.DataFrame) -> pd.DataFrame:
    """
    shelf_life is messy free text: '3-5 years', '6-12 months (frozen)',
    'Indefinite', '1-2 days fresh / ~1 yr frozen', etc.

    Takes the first numeric range + unit found as the primary shelf-life
    estimate (in days). Rows that say "Indefinite" or don't parse cleanly
    are flagged via is_shelf_life_indefinite rather than guessing a number --
    an indefinite product and a genuinely-unparsed string shouldn't collapse
    into the same numeric value.
    """
    pattern = re.compile(r"(\d+\.?\d*)\s*-?\s*(\d+\.?\d*)?\s*(day|days|week|weeks|month|months|year|years|yrs)")

    min_days, max_days, is_indef = [], [], []
    for val in master["shelf_life"]:
        if pd.isna(val):
            min_days.append(np.nan); max_days.append(np.nan); is_indef.append(False)
            continue
        if "indefinite" in val.lower():
            min_days.append(np.nan); max_days.append(np.nan); is_indef.append(True)
            continue
        m = pattern.search(val)
        if not m:
            min_days.append(np.nan); max_days.append(np.nan); is_indef.append(False)
            continue
        lo, hi, unit = m.groups()
        unit_days = _SHELF_LIFE_UNIT_DAYS[unit]
        lo_days = float(lo) * unit_days
        hi_days = float(hi) * unit_days if hi else lo_days
        min_days.append(lo_days); max_days.append(hi_days); is_indef.append(False)

    master["shelf_life_days_min"] = min_days
    master["shelf_life_days_max"] = max_days
    master["is_shelf_life_indefinite"] = is_indef
    return master


def load_daily_panel(path: str = DAILY_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def fix_christmas_gap(df: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    actual_dates = set(df["date"].unique())
    missing_dates = [d for d in full_range if d not in actual_dates]

    if not missing_dates:
        return df

    store_family = df[["store_nbr", "family", "city", "state", "type", "cluster"]].drop_duplicates()

    fill_rows = []
    for d in missing_dates:
        block = store_family.copy()
        block["date"] = d
        block["sales"] = 0.0
        block["onpromotion"] = 0
        block["is_holiday"] = True
        block["oil_price"] = np.nan
        fill_rows.append(block)

    fill_df = pd.concat(fill_rows, ignore_index=True)
    fill_df["id"] = np.arange(df["id"].max() + 1, df["id"].max() + 1 + len(fill_df))

    out = pd.concat([df, fill_df[df.columns]], ignore_index=True)
    out = out.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return out


def fill_oil_price(df: pd.DataFrame) -> pd.DataFrame:
    oil_by_date = (
        df[["date", "oil_price"]]
        .drop_duplicates(subset="date")
        .sort_values("date")
    )
    oil_by_date["oil_price"] = oil_by_date["oil_price"].ffill().bfill()
    df = df.drop(columns=["oil_price"]).merge(oil_by_date, on="date", how="left")
    return df


def load_and_prepare_master(path: str = MASTER_PATH) -> pd.DataFrame:
    master = pd.read_csv(path)
    master = parse_price_range(master)
    master = parse_shelf_life(master)
    drop_cols = [c for c in MASTER_DROP_COLS if c in master.columns]
    master = master.drop(columns=drop_cols)
    return master


def join_static_features(daily: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    return daily.merge(master, on=["store_nbr", "family"], how="left", validate="many_to_one")


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 28),
) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    grp = df.groupby(["store_nbr", "family"], sort=False)["sales"]

    for lag in lags:
        df[f"sales_lag_{lag}"] = grp.shift(lag)

    for window in rolling_windows:
        shifted = grp.shift(1)
        df[f"sales_roll_mean_{window}"] = (
            shifted.groupby([df["store_nbr"], df["family"]]).transform(
                lambda s: s.rolling(window, min_periods=max(3, window // 4)).mean()
            )
        )
        df[f"sales_roll_std_{window}"] = (
            shifted.groupby([df["store_nbr"], df["family"]]).transform(
                lambda s: s.rolling(window, min_periods=max(3, window // 4)).std()
            )
        )

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["year"] = df["date"].dt.year
    df["is_month_start"] = df["date"].dt.is_month_start
    df["is_month_end"] = df["date"].dt.is_month_end

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in (DAILY_CATEGORICAL_COLS + STATIC_CATEGORICAL_COLS) if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


def assert_no_raw_object_columns(df: pd.DataFrame):
    """Fail loudly here rather than let LightGBM fail cryptically later."""
    bad = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    if bad:
        raise TypeError(
            f"These columns are still raw text and will break LightGBM: {bad}. "
            f"Cast to category, parse to numeric, or drop them."
        )


def build_training_table() -> pd.DataFrame:
    t0 = time.time()
    daily = load_daily_panel()
    print(f"[t={time.time()-t0:.1f}s] daily panel loaded: {daily.shape}")

    daily = fix_christmas_gap(daily)
    print(f"[t={time.time()-t0:.1f}s] christmas gap fixed: {daily.shape}")

    daily = fill_oil_price(daily)
    print(f"[t={time.time()-t0:.1f}s] oil price filled, nulls remaining: {daily['oil_price'].isna().sum()}")

    master = load_and_prepare_master()
    print(f"[t={time.time()-t0:.1f}s] master loaded: {master.shape}")

    merged = join_static_features(daily, master)
    print(f"[t={time.time()-t0:.1f}s] joined: {merged.shape}")

    merged = add_lag_and_rolling_features(merged)
    print(f"[t={time.time()-t0:.1f}s] lag/rolling features added: {merged.shape}")

    merged = encode_categoricals(merged)
    assert_no_raw_object_columns(merged)
    print("No raw text columns remain -- safe for LightGBM. DONE.")

    return merged


def save_training_table(df: pd.DataFrame, path_stem: str = "training_table") -> str:
    """Saves to parquet if pyarrow is available, otherwise CSV. Returns the actual path written."""
    try:
        path = f"{path_stem}.parquet"
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        path = f"{path_stem}.csv"
        df.to_csv(path, index=False)
        return path


def load_training_table(path_stem: str = "training_table") -> pd.DataFrame:
    """
    Loads the pre-built merged table. train_lightgbm.py, train_arima.py, and
    train_prophet.py call this -- NOT build_training_table() directly -- so
    the join/feature-engineering logic runs once (via `python build_training_set.py`)
    rather than being repeated inside every model script.
    """
    parquet_path = f"{path_stem}.parquet"
    csv_path = f"{path_stem}.csv"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        # hier_cluster/volume_cluster mix sentinel strings ("NO_SALES",
        # "INSUFFICIENT_DATA") with normal labels in a way that can trip
        # pandas' chunked type inference on a CSV this large -- force just
        # those two columns rather than the whole file (forcing everything
        # via low_memory=False spikes peak memory unnecessarily).
        df = pd.read_csv(
            csv_path, parse_dates=["date"],
            dtype={"hier_cluster": str, "volume_cluster": str},
        )
        return encode_categoricals(df)
    else:
        raise FileNotFoundError(
            f"No training table found at {parquet_path} or {csv_path}. "
            f"Run `python build_training_set.py` first to build it."
        )


def walk_forward_splits(dates: pd.Series, n_splits: int = 5, horizon: int = 15, gap: int = 0):
    """
    Expanding-window walk-forward splits.

    dates:    the full sorted set of unique dates in the training table
    n_splits: number of folds
    horizon:  forecast length per fold -- match your real production horizon
    gap:      buffer days between train end and val start (>0 if long
              rolling-window features could otherwise leak across the boundary)

    Returns a list of (train_dates, val_dates) in chronological order.
    """
    unique_dates = sorted(pd.Series(dates).unique())
    splits = []
    for i in range(n_splits):
        val_end_idx = len(unique_dates) - i * horizon
        val_start_idx = val_end_idx - horizon
        train_end_idx = val_start_idx - gap
        if train_end_idx <= 0:
            break
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]
        splits.append((train_dates, val_dates))
    return list(reversed(splits))


if __name__ == "__main__":
    table = build_training_table()
    print("\nFinal shape:", table.shape)
    print("Memory usage (MB):", table.memory_usage(deep=True).sum() / 1e6)

    saved_path = save_training_table(table)
    print(f"\nSaved merged training table to {saved_path}")