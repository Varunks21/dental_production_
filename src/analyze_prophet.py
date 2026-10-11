"""Use Prophet per practice to summarize long-term trend and monthly seasonality.

This script is descriptive: it fits each practice's full history and extracts
trend/month effects. Its in-sample fitted values MUST NOT be reported as an
out-of-sample forecast accuracy score. See README for interpretation limits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

from common import KEY_COLUMNS, TARGET, load_data

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fit_practice(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    
    practice_key = group["practice_key"].iloc[0]
    history = group[["date", TARGET]].rename(columns={"date": "ds", TARGET: "y"}).copy()
    history["ds"] = pd.to_datetime(history["ds"])
    for month in range(2, 13):  # January is the reference month
        history[f"month_{month:02d}"] = (history["ds"].dt.month == month).astype(int)

    model = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        interval_width=0.80,
    )
    for month in range(2, 13):
        model.add_regressor(f"month_{month:02d}", standardize=False)
    model.fit(history)

    forecast = model.predict(history.drop(columns="y"))
    result = group[["date"] + KEY_COLUMNS + ["practice_key", TARGET]].copy()
    result = result.rename(columns={"date": "ds", TARGET: "actual_total_amount"})
    result["prophet_fitted"] = forecast["yhat"].to_numpy()
    result["trend"] = forecast["trend"].to_numpy()
    # With other Prophet seasonal/holiday components disabled, yhat minus trend
    # is the fitted month-indicator contribution. January is the reference level,
    # so this raw contribution is relative to January, not to the average month.
    result["monthly_effect_vs_january"] = result["prophet_fitted"] - result["trend"]

    # Center the month effects for an interpretable chart: positive means above
    # the clinic's average estimated month effect; negative means below it.
    # There are 12 observations for every calendar month in each complete series.
    mean_month_effect = result["monthly_effect_vs_january"].mean()
    result["monthly_seasonal_effect"] = (
        result["monthly_effect_vs_january"] - mean_month_effect
    )
    mean_y = result["actual_total_amount"].mean()
    denom = mean_y if abs(mean_y) > 1e-12 else np.nan
    result["monthly_seasonal_effect_pct_of_practice_mean"] = (
        result["monthly_seasonal_effect"] / denom * 100
    )

    trend_start = float(result["trend"].iloc[0])
    trend_end = float(result["trend"].iloc[-1])
    trend_change_amount = trend_end - trend_start
    trend_change_pct = ((trend_change_amount / abs(trend_start)) * 100) if abs(trend_start) > 1e-12 else np.nan
    trend_change_pct_of_mean = (
        trend_change_amount / mean_y * 100 if abs(mean_y) > 1e-12 else np.nan
    )
    seasonal_by_month = result.assign(month=result["ds"].dt.month).groupby("month")[
        "monthly_seasonal_effect_pct_of_practice_mean"
    ].mean()
    seasonality_amplitude = float(seasonal_by_month.max() - seasonal_by_month.min())
    summary = {
        "practice_key": practice_key,
        "masterid": group["masterid"].iloc[0],
        "practiceid": group["practiceid"].iloc[0],
        "custid": group["custid"].iloc[0],
        "observations": len(group),
        "start_date": group["date"].min().date().isoformat(),
        "end_date": group["date"].max().date().isoformat(),
        "prophet_trend_start": trend_start,
        "prophet_trend_end": trend_end,
        "prophet_trend_change_amount": trend_change_amount,
        "prophet_trend_change_pct": trend_change_pct,
        "prophet_trend_change_pct_of_mean_production": trend_change_pct_of_mean,
        "seasonality_amplitude_pct_of_practice_mean": seasonality_amplitude,
        "mean_observed_production": float(group[TARGET].mean()),
    }
    return result, summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project_root / "data" / "synthetic_production_data.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    components: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for _, group in df.groupby("practice_key", sort=True):
        practice_components, summary = fit_practice(group.sort_values("date"))
        components.append(practice_components)
        summaries.append(summary)

    all_components = pd.concat(components, ignore_index=True)
    trend_summary = pd.DataFrame(summaries).sort_values("prophet_trend_change_pct")
    all_components.to_csv(args.output_dir / "prophet_components_by_practice.csv", index=False, date_format="%Y-%m-%d")
    trend_summary.to_csv(args.output_dir / "prophet_practice_trend_summary.csv", index=False)

    # Reduce to one modeled effect per practice-calendar-month before
    # aggregating across practices; this avoids counting the same fixed effect
    # once for every year of history.
    practice_month_effects = (
        all_components.assign(month=all_components["ds"].dt.month)
        .groupby(["practice_key", "month"], as_index=False)[
            "monthly_seasonal_effect_pct_of_practice_mean"
        ].mean()
    )
    month_summary = (
        practice_month_effects.groupby("month")["monthly_seasonal_effect_pct_of_practice_mean"]
        .agg(mean_effect_pct="mean", median_effect_pct="median", std_effect_pct="std")
        .reset_index()
    )
    positive_share = practice_month_effects.groupby("month")[
        "monthly_seasonal_effect_pct_of_practice_mean"
    ].apply(lambda s: float((s > 0).mean())).rename("share_practice_month_effect_positive").reset_index()
    month_summary = month_summary.merge(positive_share, on="month")
    month_summary["month_name"] = month_summary["month"].map({i + 1: n for i, n in enumerate(MONTH_NAMES)})
    month_summary.to_csv(args.output_dir / "prophet_monthly_seasonality_summary.csv", index=False)

    # Descriptive aggregate history: average monthly record value across clinics by year.
    annual = df.assign(year=df["date"].dt.year).groupby("year")[TARGET].agg(
        mean_monthly_production="mean", median_monthly_production="median", total_observed_production="sum"
    ).reset_index()
    annual.to_csv(args.output_dir / "annual_production_summary.csv", index=False)

    # Plot annual average monthly production over time.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(annual["year"], annual["mean_monthly_production"], marker="o")
    ax.set_title("Average monthly production by year across practices")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean total_amount (dataset units)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "annual_average_production.png", dpi=160)
    plt.close(fig)

    # Aggregate modeled month effects (scaled to each practice's own average).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        month_summary["month_name"],
        month_summary["mean_effect_pct"],
        yerr=month_summary["std_effect_pct"],
        capsize=3,
    )
    ax.axhline(0, linewidth=1, color="black")
    ax.set_title("Prophet-estimated monthly effects across practices")
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Centered seasonal effect (% of practice mean production)")
    ax.text(
        0.01, 0.02,
        "Positive = above the clinic's average month effect; negative = below. Error bars show SD across practices.",
        transform=ax.transAxes, fontsize=8, va="bottom",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "prophet_monthly_seasonality.png", dpi=160)
    plt.close(fig)

    # Select example clinics across the observed production range rather than
    # taking the first four identifiers alphabetically. Overall summaries still
    # use all practices.
    practice_means = (
        df.groupby("practice_key")[TARGET].mean().sort_values()
    )
    selected_keys: list[str] = []
    for q in [0.10, 0.35, 0.65, 0.90]:
        target_mean = practice_means.quantile(q)
        candidate = (practice_means - target_mean).abs().idxmin()
        if candidate not in selected_keys:
            selected_keys.append(candidate)
    for candidate in practice_means.index:
        if len(selected_keys) == 4:
            break
        if candidate not in selected_keys:
            selected_keys.append(candidate)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for ax, key in zip(axes.ravel(), selected_keys):
        g = all_components[all_components["practice_key"] == key].sort_values("ds")
        mean_production = g["actual_total_amount"].mean()
        ax.plot(g["ds"], g["actual_total_amount"], label="Observed", linewidth=1.1)
        ax.plot(g["ds"], g["trend"], label="Prophet trend", linewidth=2)
        ax.set_title(f"{key}\nMean production: {mean_production:,.0f}", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("Production (dataset units)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Month")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle("Observed production and Prophet-fitted trend\nExamples spanning clinic production levels", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(plots_dir / "prophet_trend_examples.png", dpi=160)
    plt.close(fig)

    negative_fit_count = int((all_components["prophet_fitted"] < 0).sum())
    print(f"Practices analysed: {df['practice_key'].nunique()}")
    print(f"Practice-month rows analysed: {len(all_components):,}")
    print(f"Negative in-sample fitted values (diagnostic only): {negative_fit_count}")
    print("Saved Prophet trend and seasonality outputs to:", args.output_dir)
    print("Reminder: these are in-sample descriptive components, not out-of-sample accuracy results.")


if __name__ == "__main__":
    main()
