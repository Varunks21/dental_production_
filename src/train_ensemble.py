"""Train/evaluate Random Forest, XGBoost, their simple average, and baselines.

Forecast design: rolling one-month-ahead. Each target month's features use
only prior months. After a month completes, its actual results may be used for
the following month's forecast.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from common import KEY_COLUMNS, TARGET, get_feature_columns, load_data, make_features, metric_dict


ENSEMBLE_WEIGHTS = {"random_forest": 0.5, "xgboost": 0.5}


def build_pipeline(model_name: str, categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    if model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=350,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "xgboost":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=450,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=5.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])


def predict_models(train: pd.DataFrame, target_rows: pd.DataFrame,
                   feature_cols: list[str], categorical: list[str], numeric: list[str]) -> dict[str, tuple[np.ndarray, Pipeline]]:
    outputs: dict[str, tuple[np.ndarray, Pipeline]] = {}
    for name in ["random_forest", "xgboost"]:
        pipe = build_pipeline(name, categorical, numeric)
        pipe.fit(train[feature_cols], train[TARGET])
        outputs[name] = (pipe.predict(target_rows[feature_cols]), pipe)
    rf = outputs["random_forest"][0]
    xgb = outputs["xgboost"][0]
    ensemble = ENSEMBLE_WEIGHTS["random_forest"] * rf + ENSEMBLE_WEIGHTS["xgboost"] * xgb
    outputs["ensemble"] = (ensemble, None)  # type: ignore[assignment]
    return outputs


def save_feature_importance(pipelines: dict[str, Pipeline], output_path: Path) -> None:
    frames = []
    for model_name, pipeline in pipelines.items():
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        values = pipeline.named_steps["regressor"].feature_importances_
        frames.append(pd.DataFrame({"feature": names, model_name: values}))
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="feature", how="outer")
    combined = combined.fillna(0.0)
    combined["mean_importance"] = combined[["random_forest", "xgboost"]].mean(axis=1)
    combined.sort_values("mean_importance", ascending=False).to_csv(output_path, index=False)


def plot_test_predictions(pred_df: pd.DataFrame, output_dir: Path) -> None:
    # Aggregate across clinics by month. This compares totals and is not a practice-level chart.
    monthly = pred_df.groupby("date", as_index=False)[
        ["actual_total_amount", "random_forest_prediction", "xgboost_prediction", "ensemble_prediction"]
    ].sum()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col, label in [
        ("actual_total_amount", "Actual"),
        ("random_forest_prediction", "Random Forest"),
        ("xgboost_prediction", "XGBoost"),
        ("ensemble_prediction", "50/50 Ensemble"),
    ]:
        ax.plot(monthly["date"], monthly[col], marker="o", linewidth=1.8, label=label)
    ax.set_title("2024 monthly production: actual vs model predictions")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total production across practices (dataset units)")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "test_2024_monthly_actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project_root / "data" / "synthetic_production_data.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    raw = load_data(args.data)
    featured = make_features(raw)
    categorical, numeric = get_feature_columns()
    feature_cols = categorical + numeric
    # Remove rows that do not have sufficient history for all engineered inputs.
    eligible = featured.dropna(subset=numeric).copy()

    metrics_rows: list[dict[str, object]] = []
    for year in [2022, 2023]:
        train = eligible[eligible["date"] < pd.Timestamp(f"{year}-01-01")]
        valid = eligible[eligible["date"].dt.year == year]
        if train.empty or valid.empty:
            raise ValueError(f"Cannot create validation split for {year}.")
        predictions = predict_models(train, valid, feature_cols, categorical, numeric)
        for name, (pred, _) in predictions.items():
            metrics_rows.append({
                "split": "validation", "year": year, "model": name,
                **metric_dict(valid[TARGET], pred),
            })
        for name, feature in [("last_month_baseline", "revenue_lag_1"),
                              ("same_month_last_year_baseline", "revenue_lag_12")]:
            metrics_rows.append({
                "split": "validation", "year": year, "model": name,
                **metric_dict(valid[TARGET], valid[feature]),
            })

    # Final period is trained through 2023 and evaluated on 2024.
    train = eligible[eligible["date"] < pd.Timestamp("2024-01-01")]
    test = eligible[eligible["date"].dt.year == 2024]
    if train.empty or len(test) == 0:
        raise ValueError("Expected training records before 2024 and test records in 2024.")
    final_predictions = predict_models(train, test, feature_cols, categorical, numeric)
    for name, (pred, _) in final_predictions.items():
        metrics_rows.append({"split": "test", "year": 2024, "model": name,
                             **metric_dict(test[TARGET], pred)})
    for name, feature in [("last_month_baseline", "revenue_lag_1"),
                          ("same_month_last_year_baseline", "revenue_lag_12")]:
        metrics_rows.append({"split": "test", "year": 2024, "model": name,
                             **metric_dict(test[TARGET], test[feature])})

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "ensemble_metrics.csv", index=False)

    pred = test[["date"] + KEY_COLUMNS + ["practice_key", TARGET, "revenue_lag_1", "revenue_lag_12"]].copy()
    pred = pred.rename(columns={TARGET: "actual_total_amount",
                                "revenue_lag_1": "last_month_baseline",
                                "revenue_lag_12": "same_month_last_year_baseline"})
    for name in ["random_forest", "xgboost", "ensemble"]:
        pred[f"{name}_prediction"] = final_predictions[name][0]
        pred[f"{name}_residual"] = pred["actual_total_amount"] - pred[f"{name}_prediction"]
        pred[f"{name}_absolute_error"] = pred[f"{name}_residual"].abs()
    pred.to_csv(args.output_dir / "predictions_2024.csv", index=False, date_format="%Y-%m-%d")

    practice_rows = []
    for key, group in pred.groupby("practice_key"):
        row: dict[str, object] = {"practice_key": key, "n": len(group)}
        for name in ["random_forest", "xgboost", "ensemble"]:
            row.update({f"{name}_{metric}": value for metric, value in metric_dict(
                group["actual_total_amount"], group[f"{name}_prediction"]
            ).items() if metric in ["MAE", "RMSE", "WAPE_pct", "R2", "mean_error_actual_minus_predicted"]})
        practice_rows.append(row)
    pd.DataFrame(practice_rows).sort_values("ensemble_WAPE_pct").to_csv(
        args.output_dir / "practice_level_metrics_2024.csv", index=False
    )

    final_pipelines = {name: final_predictions[name][1] for name in ["random_forest", "xgboost"]}
    save_feature_importance(final_pipelines, args.output_dir / "feature_importance.csv")
    bundle = {
        "models": final_pipelines,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "feature_columns": feature_cols,
        "forecasting_assumption": "rolling one-month-ahead; prior-month actuals are available for later forecasts",
    }
    joblib.dump(bundle, args.output_dir / "trained_ensemble.joblib")
    plot_test_predictions(pred, args.output_dir)

    print(f"Practices: {raw['practice_key'].nunique()}")
    print(f"Raw records: {len(raw):,}")
    print(f"Eligible records after historical feature creation: {len(eligible):,}")
    print("\nMetrics (lower is better for MAE/RMSE/WAPE; higher for R2):")
    show = metrics.copy()
    for col in ["MAE", "RMSE", "WAPE_pct", "R2"]:
        show[col] = show[col].map(lambda x: f"{x:.4f}")
    print(show.to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
