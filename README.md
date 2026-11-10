# PTSFC 2026/27 forecasting workspace

Start here for weekly submissions, saved-result diagnostics, and controlled
weather experiments. See [the workspace review](docs/WORKSPACE_REVIEW.md) for
what was fixed and the limitations of the historical experiment report.

## Setup and weekly submission

Use the project's Python environment, then:

```powershell
python -m pip install -r requirements.txt
# Set team_name and optional forecast_date in config.py.
python run_pipeline.py
```

Run commands from the repository root. `config.py` controls data refresh,
training windows, interval spreads, backtesting and plots. Set `offline=True`
to reuse cached observations. The pipeline writes and checks the baseline
submission, copies it to `archive/`, and optionally evaluates and plots the
last 52 forecast Wednesdays.

Only rows labelled `model` go into the submission. Enhanced energy forecasts
are evaluated alongside the baseline; experimental weather variants run
separately. Neither is automatically promoted to the submission.

## Weather experiment → audit → PDF

To reproduce the period used by the original report:

```powershell
python experiment_weather.py --date 2026-11-04 --weeks 52
python audit_weather.py
python build_experiment_report.py --energy output/backtest_20261104.csv
```

Experiments and audits use cached data by default; `experiment_weather.py
--online` refreshes weather observations. Reproducing old scores exactly also
requires the same observation cache and configuration. The report reads saved
predictions and the training audit and rebuilds its summary tables, plots and
paired comparisons. Its original research conclusions and energy numbers are
explicitly labelled historical interpretation.

For a fresh run, preserve earlier results by using a separate directory:

```powershell
python experiment_weather.py --date 2026-11-11 --output output/experiments/20261111/weather
python audit_weather.py --input output/experiments/20261111/weather/predictions.csv --output output/experiments/20261111/audit
python build_experiment_report.py --predictions output/experiments/20261111/weather/predictions.csv --audit-dir output/experiments/20261111/audit --energy output/backtest_20261111.csv
```

`--output` on the report changes the PDF path; generated CSVs and figures go
to `--audit-dir`. All commands support `--help`. Generation writes `run.json`
with configuration, variant mapping, date and generation time.

| Model | Training sample | Predictors |
|---|---|---|
| W0 | Seasonal window | One daily lag (baseline) |
| W1 | Seasonal window | Two daily lags |
| W2a | Full historical window | Two daily lags |
| W2b | Full historical window | Two daily lags + annual sine/cosine |
| W2c | Full historical window | One daily lag + annual sine/cosine |

All use OLS plus empirical residual quantiles. Weather wind quantiles are
clipped at zero. Compare matched observations and weekly mean quantile scores
(lower is better), alongside 50% and 95% coverage. Bootstrap intervals are
exploratory because configurations were inspected on the same evaluation year.

## Inspect saved forecasts

```powershell
python diagnostics.py --input output/backtest_20261111.csv
python forecast_fan.py --date 2026-11-11
```

Diagnostics evaluate saved baseline rows. Fan plots read the saved forecasts
and cached observations; add `--show` for an interactive window. Historical
defaults remain November 4. Outputs go to `output/diagnostics/`; general
diagnostic plots are overwritten by the next run, while fan evaluation tables
and figures include their forecast date.

## Where things belong

| Location | Purpose |
|---|---|
| `config.py` | Shared forecast parameters and submission constants |
| `run_pipeline.py`, `check_submission.*` | Submission workflow and format checks |
| `ptsfc/models.py` | Baseline and candidate forecast functions |
| `ptsfc/backtest.py`, `ptsfc/utils.py` | Evaluation, scores, dates and regression helpers |
| `ptsfc/data_*.py`, `ptsfc/plots.py` | Data loading and standard plots |
| `experiment_weather.py`, `audit_weather.py`, `build_experiment_report.py` | Research workflow |
| `diagnostics.py`, `forecast_fan.py` | Saved-result analysis |
| `data/` | Local SMARD/DWD observation caches |
| `output/` | Generated submissions, backtests, plots, experiments and reports |
| `archive/` | Submission history; retain deliberately |
| `docs/` | Review notes and methodological limitations |

New caches, outputs, bytecode, IDE state and the virtual environment are
ignored by Git. Existing tracked artifacts remain tracked. Archive submissions
remain visible so they can be committed deliberately. No old results need
to be deleted to use this structure.

For each future change, record the hypothesis, configuration, Git revision,
data acquisition time, matched scores and coverage. Change one element at a
time and validate on fresh forecast weeks before adopting it.
