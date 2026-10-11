# Prophet output interpretation

The included Prophet outputs summarize fitted trends and estimated calendar-month effects across all 56 practices. To regenerate them after installing `requirements.txt`, run:

```bash
python src/analyze_prophet.py
```

## Interpretation notes

- `prophet_components_by_practice.csv` contains observed production, in-sample fitted production, fitted trend, the month-indicator contribution relative to January, and a centered seasonal effect.
- January is the reference category for the month-indicator regressors, so its raw contribution is zero by construction. Do not interpret that raw zero as an absence of seasonality. The centered `monthly_seasonal_effect` subtracts the practice-specific average over the twelve month effects.
- `monthly_seasonal_effect_pct_of_practice_mean` is the centered effect divided by that practice's observed mean production. Positive values mean above-average estimated monthly seasonality; negative values mean below-average.
- The aggregate month chart shows mean centered effects; error bars show standard deviation across practices. The pattern describes this supplied dataset and should not be generalized to all dental clinics.
- `prophet_practice_trend_summary.csv` reports both absolute trend change and percentage change. Percentage changes can be very large when the fitted trend at the start is small, so interpret them alongside absolute change and mean observed production.
- One or more fitted values may be negative because Prophet's additive model does not constrain predictions to nonnegative values. In this project these are in-sample descriptive fits, not deployed forecasts. If forecasts are later used operationally, assess a nonnegative target transformation or clipping strategy using validation data rather than silently altering the descriptive components.
- Fitted trends and effects use each practice's complete 2013–2024 history. They are not out-of-sample accuracy estimates. A fair Prophet forecast benchmark would require a separate chronological rolling-origin evaluation.
