# Take-home assessment write-up

## Approach

I treat the task as monthly, rolling one-month-ahead forecasting of `total_amount` for an existing dental practice, uniquely identified by `(masterid, practiceid, custid)`. After validating date/key integrity and sorting observations by practice and month, I engineer leakage-safe features: revenue lags (1, 2, 3, 6 and 12 months), trailing revenue averages and volatility, lags/three-month averages for provider count, procedure count, working days and operatories, calendar trend/seasonality, and historical productivity ratios. Every lag and rolling feature is shifted to use information available before the target month.

## Models

I compare Random Forest and XGBoost, then average their predictions with equal weights. Random Forest aggregates trees trained on varied bootstrap samples, while XGBoost builds trees sequentially to improve the existing model. Their combination is a hypothesis about error complementarity, not a guaranteed improvement. The code reports both individual models as well as the ensemble and two simple historical baselines. Ensemble weights are fixed at 50/50 rather than tuned on the test period.

## Evaluation

I use chronological validation in 2022 and 2023 and a 2024 test period. Metrics include MAE, RMSE, WAPE and R². In the included run, 2024 WAPE was 7.47% for Random Forest, 7.15% for XGBoost and 7.13% for the 50/50 ensemble. The ensemble's gain over XGBoost on the test period is very small; on 2022 validation, XGBoost outperformed the ensemble, while in 2023 the ensemble was slightly better. Thus the current evidence does not prove that the ensemble is consistently better than XGBoost alone. I also produce practice-level errors to identify where aggregate metrics might hide poor performance.

The forecasting assumption is rolling one-month-ahead: the previous month's actuals can be used for later forecasts. This differs from predicting an entire future year at one time.

## Prophet analysis

Separately, `src/analyze_prophet.py` fits Prophet per practice to summarize fitted trends and calendar-month effects across the historical data. Because each observation is monthly, the script uses month-of-year indicator regressors instead of fitting intra-month seasonal detail. January is the reference category in the regression; for more interpretable reporting, the script centers each practice's twelve month effects so positive/negative values mean above/below that practice's average monthly effect. It also reports absolute trend change alongside percentage change, since percentage changes can be unstable when the initial fitted trend is near zero. The analysis is descriptive and fitted on each practice's complete history; it is not an out-of-sample forecast benchmark. In the saved components, the centered effects indicate that July and October are above the practice-average monthly seasonal contribution, while February, November and December are below it on average. These are patterns in this supplied synthetic dataset, not universal dental-industry claims. The plots and CSV summaries are included in the project, and can be reproduced with `python src/analyze_prophet.py` after installing dependencies.

## Limitations and next steps

Parameters are reasonable starting configurations, not the result of exhaustive tuning. I would add robust checks for unusual values, compare more validation origins, inspect residuals and feature importance, test different ensemble weights using validation only, and evaluate a later untouched period if available. Since the CSV is named synthetic, observed patterns should not be assumed to generalize to the entire dental industry.
