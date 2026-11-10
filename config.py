"""
PTSFC 2026/27 - BASELINE CONFIGURATION
Deliberately small: a simple starting point to improve week by week (see README, "Roadmap").
"""

CFG = dict(
    # ----------------------------------------------------------- General --
    team_name="MonteCarloMafia",  # submission filename suffix
    forecast_date=None,          # None = next Wednesday (today if Wednesday); or "2026-11-04"
    offline=False,               # True = use only the cached data in data/
    dir_data="data",
    dir_output="output",
    dir_archive="archive",       # every submission is copied here
    r_check_script="check_submission.R",  # also run the R check if Rscript is installed
    make_plots=True,
    run_backtest=True,
    backtest_weeks=52,

    # ----------------------------------------------------- Energy (SMARD) --
    energy_source="api",                           # "api" or "csv" (raw SMARD export)
    energy_csv_path=None,
    energy_csv_cols=dict(demand=3, res_demand=6),
    smard_filters=dict(demand=410, res_demand=4359),
    smard_resolution="hour",
    energy_start_date="2020-01-01",

    energy_train_weeks=104,      # training window: last N same-weekday observations
    energy_spread=dict(demand=1.25, res_demand=1.2),   # interval width multiplier

    # ------------------------------------------------- Weather (DWD 1420) --
    dwd_station_id=1420,
    weather_train_years=10,      # years of history used for training
    weather_season_days=30,      # only training days within +/- N days of the target's day of year
    weather_spread=dict(temperature=1.0, wind=1.0),

    # Calendar days removed from TRAINING (holidays + Christmas/New Year period)
    drop_holidays_from_training=True,
)

# Submission format constants (do not change)
QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]
QCOLS = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
HORIZONS = [36, 40, 44, 60, 64, 68]
TARGETS = ["demand", "res_demand", "temperature", "wind"]
