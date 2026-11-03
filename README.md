# PTSFC 2026/27 – Baseline pipeline (simple starting point)

Deliberately simple models, designed to be improved one step at a time.
The pipeline downloads the data, fits the models, writes the submission CSV,
runs the course check (Python port + the original R script if `Rscript` exists),
runs a 52-week backtest and saves a few plots.

```bash
pip install -r requirements.txt
# set team_name in config.py, then:
python run_pipeline.py
```

The `data/` cache from the previous version is compatible (same file names), so nothing is downloaded again.

## The models

**Energy (demand, res_demand)** – one multiple linear regression per target slot
(Fri/Sat × hour ending 12/16/20 German time):

    y(day, h) = b0 + b1 · y(last complete day, h) + b2 · y(same slot one week earlier) + e

Trained on the last `energy_train_weeks` observations of the same weekday and hour.
Holidays and the Christmas/New Year period are removed from the training data (no holiday dummies).

**Weather (temperature, wind)** – one regression with a lagged variable per horizon:

    y(T) = b0 + b1 · y(last observed day, same clock hour) + e

Trained only on days of the same season (± `weather_season_days` around the target's day of year,
over `weather_train_years` years). The seasonal window replaces explicit seasonality terms.
Wind quantiles are clipped at 0.

**Uncertainty (all targets)** – point forecast + `spread` × empirical quantiles of the regression residuals.

**References in the backtest**: the course benchmark for energy (last 20 same weekday/hour values)
and the seasonal climatology for weather (same training window, no regression).

## Files

```
config.py            all parameters
run_pipeline.py      main script
ptsfc/models.py      THE models (~100 lines) – this is the file to improve week by week
ptsfc/backtest.py    re-runs the same forecast for the last N Wednesdays and scores it
ptsfc/plots.py       preprocessing, calibration and current-forecast plots
ptsfc/data_*.py      SMARD and DWD download + cache
ptsfc/utils.py       dates, holidays, OLS + residual quantiles, quantile score
check_submission.*   course check (Python port and original R)
```

## How to improve week by week

1. Change **one thing** at a time.
2. Run the pipeline and compare the backtest summary (`mean_QS`, `cov50`, `cov95`) with the previous version.
3. Keep the change only if the mean quantile score drops (and calibration does not get worse).
4. Write the change and the scores in a changelog (and commit it to git).

## Roadmap – candidate improvements (rough order of expected gain / effort)

| # | Improvement | Target | Why | Effort |
|---|---|---|---|---|
| 1 | Tune `*_spread` from backtest coverage: `new = old · 1.96 / z`, with `z = Φ⁻¹((1 + cov95)/2)` | all | Intervals from in-sample residuals tend to be too narrow | trivial |
| 2 | Use the ECMWF ensemble (Open-Meteo): raw quantiles first, then bias + spread correction, then blend with this baseline | temperature, wind | Lags carry little information 3 days ahead; the course benchmark is the ensemble | medium |
| 3 | Add wind and solar forecasts as regressors | res_demand | The weather drives most of the residual load | medium |
| 4 | Season-dependent residuals (e.g. only residuals from ± 45 days of year) | demand, res_demand | Errors are larger in winter (cold spells, wind) | small |
| 5 | Annual seasonality terms (Fourier sin/cos of day of year) or a seasonal training window | demand, res_demand | Tuesday→Friday relation changes over the year | small |
| 6 | Temperature as a regressor (heating demand) | demand | Explains winter peaks | medium |
| 7 | Holiday dummies (only weekday holidays, minimum number of cases) instead of dropping days | demand, res_demand | Needed for holiday weeks after the Christmas break | small |
| 8 | sqrt transform for wind (fit on sqrt scale, square the quantiles) | wind | Skewed, non-negative variable | small |
| 9 | Quantile regression instead of OLS + residuals | all | Lets the interval width depend on the regressors | small |
| 10 | Evaluate the archived submissions every week (scores vs. benchmark) | all | Real out-of-sample feedback, needed to tune the ensemble | small |
| 11 | Combine several models (quantile averaging), weights from the backtest/archive | all | Combinations are usually more robust | medium |
