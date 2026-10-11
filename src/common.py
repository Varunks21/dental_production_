"""Shared loading, validation, feature engineering, and metric helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

KEY_COLUMNS = ["masterid", "practiceid", "custid"]
TARGET = "total_amount"
OPERATIONAL_COLUMNS = [
    "provider_count",
    "procedure_count",
    "working_days",
    "operatory_count",
]
REVENUE_LAGS = [1, 2, 3, 6, 12]
REVENUE_ROLLING_WINDOWS = [3, 6, 12]
OPERATIONAL_LAGS = [1, 3, 12]


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and validate practice-month keys, target, and required fields."""
    df = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["date", TARGET] + OPERATIONAL_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df[KEY_COLUMNS + ["date"]].isna().any().any():
        raise ValueError("Practice identifiers and date must not contain nulls.")
    if df[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values.")
    if not np.isfinite(pd.to_numeric(df[TARGET], errors="coerce")).all():
        raise ValueError(f"{TARGET} contains non-numeric or non-finite values.")
    if df.duplicated(KEY_COLUMNS + ["date"]).any():
        raise ValueError("Duplicate records found for a practice and month.")

    df["practice_key"] = df[KEY_COLUMNS].astype(str).agg("|".join, axis=1)
    df = df.sort_values(["practice_key", "date"]).reset_index(drop=True)

    # This project expects monthly observations. Validate calendar regularity so
    # shift(1) represents the immediately preceding calendar month.
    for practice, group in df.groupby("practice_key", sort=False):
        months = group["date"].dt.to_period("M")
        if months.duplicated().any():
            raise ValueError(f"Duplicate calendar month detected for practice {practice}.")
        expected = pd.period_range(months.min(), months.max(), freq="M")
        if len(expected) != len(months) or not np.array_equal(expected.asi8, months.array.asi8):
            raise ValueError(
                f"Practice {practice} does not have a complete monthly sequence. "
                "Resolve missing months before using row-based lag features."
            )

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged/rolling features from prior months only (no target leakage)."""
    out = df.copy()
    first_date = out["date"].min()
    out["time_index"] = (
        (out["date"].dt.year - first_date.year) * 12
        + (out["date"].dt.month - first_date.month)
    ).astype(float)
    month = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["quarter"] = out["date"].dt.quarter.astype(str)

    grouped = out.groupby("practice_key", sort=False)
    for lag in REVENUE_LAGS:
        out[f"revenue_lag_{lag}"] = grouped[TARGET].shift(lag)
    for window in REVENUE_ROLLING_WINDOWS:
        out[f"revenue_rollmean_{window}"] = grouped[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
        )
        out[f"revenue_rollstd_{window}"] = grouped[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).std()
        )

    for col in OPERATIONAL_COLUMNS:
        grouped = out.groupby("practice_key", sort=False)
        for lag in OPERATIONAL_LAGS:
            out[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
        out[f"{col}_rollmean_3"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=3).mean()
        )

    out["revenue_mom_change"] = out["revenue_lag_1"] - out["revenue_lag_2"]
    out["revenue_yoy_change"] = out["revenue_lag_1"] - out["revenue_lag_12"]
    # Historical, leakage-safe productivity ratios. Ratios use lagged values only.
    safe_procedure = out["procedure_count_lag_1"].replace(0, np.nan)
    safe_provider = out["provider_count_lag_1"].replace(0, np.nan)
    safe_days = out["working_days_lag_1"].replace(0, np.nan)
    safe_rooms = out["operatory_count_lag_1"].replace(0, np.nan)
    out["historical_production_per_procedure"] = out["revenue_lag_1"] / safe_procedure
    out["historical_procedures_per_provider"] = out["procedure_count_lag_1"] / safe_provider
    out["historical_production_per_working_day"] = out["revenue_lag_1"] / safe_days
    out["historical_procedures_per_operatory"] = out["procedure_count_lag_1"] / safe_rooms
    return out


def get_feature_columns() -> tuple[list[str], list[str]]:
    categorical = ["practice_key", "masterid", "quarter"]
    numeric = ["time_index", "month_sin", "month_cos"]
    numeric += [f"revenue_lag_{lag}" for lag in REVENUE_LAGS]
    numeric += [f"revenue_rollmean_{w}" for w in REVENUE_ROLLING_WINDOWS]
    numeric += [f"revenue_rollstd_{w}" for w in REVENUE_ROLLING_WINDOWS]
    for col in OPERATIONAL_COLUMNS:
        numeric += [f"{col}_lag_{lag}" for lag in OPERATIONAL_LAGS]
        numeric += [f"{col}_rollmean_3"]
    numeric += [
        "revenue_mom_change",
        "revenue_yoy_change",
        "historical_production_per_procedure",
        "historical_procedures_per_provider",
        "historical_production_per_working_day",
        "historical_procedures_per_operatory",
    ]
    return categorical, numeric


def metric_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(actual).sum()
    return {
        "n": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "WAPE_pct": float(np.abs(actual - pred).sum() / denom * 100) if denom else float("nan"),
        "R2": float(r2_score(actual, pred)),
        "mean_error_actual_minus_predicted": float(np.mean(actual - pred)),
    }
