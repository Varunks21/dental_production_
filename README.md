# Dental Practice Production Forecasting: Random Forest + XGBoost + Prophet

This take-home project has two separate components:

1. **Forecasting experiment:** compare Random Forest, XGBoost, their equal-weight ensemble, and simple historical baselines for monthly `total_amount`.
2. **Trend/seasonality analysis:** use Prophet separately for each practice to describe long-term trend and estimated calendar-month effects across 2013–2024.

The dataset's practice identity is the full `(masterid, practiceid, custid)` combination. A row represents one practice-month.

## Setup (Python 3.10+ recommended)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Prophet uses a Stan backend. If its installation fails in your environment, follow the current instructions in the [official Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html).

## Run the forecasting experiment

```bash
python src/train_ensemble.py
```

Optional custom paths:

```bash
python src/train_ensemble.py --data data/synthetic_production_data.csv --output-dir outputs
```

Outputs include:

- `outputs/ensemble_metrics.csv` — MAE, RMSE, WAPE and R² for the two individual models, the 50/50 ensemble and two baselines on 2022/2023 validation and 2024 test.
- `outputs/predictions_2024.csv` — practice-month predictions and errors for the final evaluation period.
- `outputs/practice_level_metrics_2024.csv` — per-practice errors; useful for identifying clinics that are difficult to forecast.
- `outputs/feature_importance.csv` — model-specific feature importance from Random Forest and XGBoost. Importance describes model use, not causation.
- `outputs/trained_ensemble.joblib` — generated when you run the training script; omitted from the submission ZIP to keep it small.
- `outputs/plots/test_2024_monthly_actual_vs_predicted.png` — aggregate actual vs predicted production for 2024.

The ensemble is a simple average:

`ensemble_prediction = 0.5 * random_forest_prediction + 0.5 * xgboost_prediction`

Weights are not tuned on the test set. The script also reports individual models so the ensemble's additional value can be assessed rather than assumed.

## Run Prophet trend and seasonality analysis

```bash
python src/analyze_prophet.py
```

Outputs include:

- `outputs/prophet_components_by_practice.csv` — observed values, in-sample fitted values, trend, raw month effect relative to January, and centered month effect for every practice-month.
- `outputs/prophet_practice_trend_summary.csv` — practice-level fitted trend change and seasonal amplitude.
- `outputs/prophet_monthly_seasonality_summary.csv` — centered monthly effects normalized relative to each practice's mean production; positive means above that practice's average month effect.
- `outputs/annual_production_summary.csv` — yearly distribution summaries for the original observations.
- `outputs/plots/annual_average_production.png`
- `outputs/plots/prophet_monthly_seasonality.png`
- `outputs/plots/prophet_trend_examples.png`

For monthly data, the Prophet script uses eleven month-of-year indicator regressors with January as the reference category, and disables yearly Fourier seasonality. The raw month-indicator contribution is relative to January. For reporting, the script centers each practice's twelve estimated monthly effects so the chart shows above/below-average month effects, rather than implying January has zero seasonality. Error bars in the aggregate seasonality plot show the standard deviation across practices. The trend-example plot selects clinics spanning the observed production range and keeps its title and legend separate.

Prophet is fitted on each practice's complete history here **for descriptive trend/seasonality analysis only**. Its in-sample fitted values must not be reported as forecast accuracy. If Prophet is to be judged as a forecasting model, implement a separate chronological out-of-sample evaluation. A few fitted values can be negative because the additive model is unconstrained; inspect this diagnostic and do not use in-sample fitted values as operational forecasts.

## Evaluation design and assumptions

- **Validation:** 2022 and 2023, each predicted by a model trained on earlier eligible rows.
- **Final evaluation period in this script:** 2024, with training data ending in December 2023.
- **Forecasting setup:** rolling one-month-ahead. For example, January's actual result can be used as historical input when forecasting February. This is not a full-year forecast made all at once at the end of 2023.
- **No same-month target leakage:** all revenue and operational lags/rolling features are shifted so they use past observations only. Same-month `total_amount` is the target, not an input.
- **Early history:** first 12 months per practice are dropped because features require up to 12 months of prior history. Some ratio/standard-deviation features can also be undefined if a denominator is zero or history is incomplete.
- **No model is assumed to be best:** compare individual validation metrics, inspect practice-level errors and only retain the ensemble if it adds value. 2024 should not be repeatedly used to tune choices.

## Why these methods?

Random Forest uses bagging/averaging across trees; XGBoost uses sequential gradient boosting. Their predictions are averaged because the approaches may make partly different errors. Complementarity and improved performance are hypotheses to verify on validation data, not guarantees. Prophet is used separately to interpret trend and calendar-month patterns; it does not feed the ensemble in this implementation.

See `writeup.md` for a short assessment summary.

## Results from the run included in this ZIP

The ensemble training script was executed on the supplied CSV. For the rolling one-month-ahead 2024 test period, the saved results are:

| Model | MAE | RMSE | WAPE | R² |
|---|---:|---:|---:|---:|
| Random Forest | 14,108.19 | 25,451.25 | 7.47% | 0.9667 |
| XGBoost | 13,498.59 | 25,004.59 | 7.15% | 0.9679 |
| 50/50 ensemble | 13,462.84 | 24,669.64 | 7.13% | 0.9687 |
| Previous-month baseline | 22,499.86 | 35,286.46 | 11.91% | 0.9361 |
| Same-month-last-year baseline | 24,403.02 | 41,150.12 | 12.92% | 0.9130 |

The ensemble's improvement over XGBoost alone is small and not consistent across both validation years. The main purpose of the project is to compare approaches transparently; these results do not establish that the ensemble is universally the best model. Note that 2024 is held out from model fitting, but its scores have been inspected during project review; treat them as exploratory rather than a pristine blind benchmark for any further model selection.

The fitted `.joblib` model is intentionally excluded from the submission ZIP to keep it small; running `src/train_ensemble.py` will regenerate it. Prophet outputs in this ZIP are based on the saved fitted components from the local Prophet run, with the month-effect summaries and charts recalculated for clearer interpretation. Running `src/analyze_prophet.py` refits Prophet and regenerates the output files. See `outputs/README_Prophet.md` for interpretation limits.
