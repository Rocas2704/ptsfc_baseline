"""
PTSFC 2026/27 - CONFIGURATION (v2 + enhanced weather model)
"""

CFG = dict(
    # ----------------------------------------------------------- General --
    team_name="MonteCarloMafia",          # -> output/20261104_MyTeam.csv
    forecast_date=None,          # None = next Wednesday (today if Wednesday); or "2026-11-04"
    offline=False,               # True = use only the cached data in data/
    dir_data="data",
    dir_output="output",
    dir_archive="archive",       # every submission is copied here
    r_check_script="check_submission.R",  # also run the R check if Rscript is installed
    make_plots=True,
    run_backtest=True,
    backtest_weeks=52,
    backtest_weather_daily=True,  # weather backtest: every day as origin (not only Wednesdays)

    # Uncertainty: "loo" = leave-one-out residuals (out-of-sample) -> spreads back to 1.0
    residuals="loo",

    # ----------------------------------------------------- Energy (SMARD) --
    energy_source="api",                           # "api" or "csv" (raw SMARD export)
    energy_csv_path=None,
    energy_csv_cols=dict(demand=3, res_demand=6),
    smard_filters=dict(demand=410, res_demand=4359),
    smard_resolution="hour",
    energy_start_date="2020-01-01",

    energy_train_weeks=104,      # training window: last N same-weekday observations
    energy_spread=dict(demand=1.0, res_demand=1.0),   # interval width multiplier

    # v2: ECMWF forecast as regressors (names = keys built in data_nwp.load_energy_nwp)
    energy_use_nwp=True,
    energy_nwp_features=dict(
        demand=["temp", "hdd", "solar"],
        res_demand=["wind_power", "solar", "temp", "hdd"],
    ),
    # (name, lat, lon) averaged per group; wind = north, solar = south, load = big cities
    energy_nwp_points=dict(
        wind=[("husum", 54.48, 9.05), ("emden", 53.37, 7.21), ("rostock", 54.09, 12.10),
              ("magdeburg", 52.13, 11.62), ("cottbus", 51.76, 14.33)],
        solar=[("munich", 48.14, 11.58), ("stuttgart", 48.78, 9.18), ("nuremberg", 49.45, 11.08),
               ("leipzig", 51.34, 12.37), ("hanover", 52.37, 9.73)],
        load=[("berlin", 52.52, 13.40), ("hamburg", 53.55, 9.99), ("cologne", 50.94, 6.96),
              ("munich", 48.14, 11.58), ("frankfurt", 50.11, 8.68)],
    ),
    wind_power_curve=dict(cut_in=3.0, rated=12.0, cut_out=25.0),
    hdd_base=15.0,               # heating degrees = max(hdd_base - temp, 0)

    # ------------------------------------------------- Weather (DWD 1420) --
    dwd_station_id=1420,
    station_lat=50.0259,         # Frankfurt/Main airport
    station_lon=8.5213,
    weather_train_years=10,      # years of history used for training
    weather_season_days=30,      # only training days within +/- N days of the target's day of year
    weather_spread=dict(temperature=1.0, wind=1.0),

    # Submitted weather model: "enhanced" | "mos" | "lagreg"  (fallback: enhanced -> mos -> lagreg)
    weather_method="enhanced",
    weather_enhanced_variant="lag2",   # lag2 | full_lag2 | fourier | fourier_lag1, or a dict per variable
    # MOS (still computed for comparison)
    wind_sqrt=True,
    mos_window_days=60,          # last N days ...
    mos_season_days=30,          # ... + same season (+/- N days) in earlier years
    mos_min_rows=60,

    # ---------------------------------------------- NWP (Open-Meteo, ECMWF) --
    nwp_enabled=True,
    nwp_url="https://previous-runs-api.open-meteo.com/v1/forecast",
    nwp_model="ecmwf_ifs025",
    nwp_start_date="2024-01-01",  # archive of the Previous Runs API
    nwp_max_lead_day=3,          # Fri = d2, Sat = d3
    nwp_chunk_days=90,           # days per request
    nwp_refresh_days=7,          # re-download the last N days on every run

    # ---------------------------------------------- Paired comparisons --
    # (target, reference, candidate); "model" = what is submitted
    comparisons=[
        ("demand", "lagonly", "model"),
        ("res_demand", "lagonly", "model"),
        ("temperature", "mos", "model"),
        ("temperature", "lagreg", "model"),
        ("wind", "mos", "model"),
        ("wind", "lagreg", "model"),
    ],
    bootstrap_block_days=28,
    bootstrap_n=2000,

    # Calendar days removed from TRAINING (holidays + Christmas/New Year period)
    drop_holidays_from_training=True,
)

# Submission format constants (do not change)
QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]
QCOLS = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
HORIZONS = [36, 40, 44, 60, 64, 68]
TARGETS = ["demand", "res_demand", "temperature", "wind"]
