"""
BASELINE MODELS - kept deliberately simple.

ENERGY (demand, res_demand): one multiple linear regression per target slot
    y(day, hour) = b0 + b1 * y(last complete day, same hour) + b2 * y(same slot one week earlier) + e
    trained on the last N observations of the same weekday and hour.

WEATHER (temperature, wind): one regression with a lagged variable per horizon
    y(T) = b0 + b1 * y(last observed day, same clock hour) + e
    trained only on days of the same season (+/- N days of the year) -> seasonality without extra terms.

UNCERTAINTY (all targets): q_tau = point forecast + spread * empirical tau-quantile of the residuals.

Every forecast only uses information available on the submission Wednesday, so the same
functions are reused for the backtest (just call them with a past Wednesday).
"""
import numpy as np
import pandas as pd

from config import HORIZONS, QCOLS, QUANTILES
from .utils import is_special_day, ols_quantiles, wday_u

DAY, H = pd.Timedelta(days=1), pd.Timedelta(hours=1)

# horizon -> (days after the Wednesday, hour ending in German local time)
ENERGY_SLOTS = {36: (2, 12), 40: (2, 16), 44: (2, 20), 60: (3, 12), 64: (3, 16), 68: (3, 20)}


# =================================================================== ENERGY ==
def forecast_energy(ED, col, fd, horizon, cfg):
    day_offset, hour = ENERGY_SLOTS[horizon]
    V = ED.wide[col][hour]                              # one value per day: this target at this hour
    td = fd + day_offset * DAY                          # target day (Friday or Saturday)
    lcd = ED.last_complete_day(col, fd - DAY)           # latest complete day known on Wednesday (Tuesday)
    ref_lag = (td - lcd).days                           # Fri - Tue = 3, Sat - Tue = 4
    week_lag = 7 * int(np.ceil(ref_lag / 7))            # same slot one week earlier (7)


    #Training data_set
    # Training rows: same weekday as the target, already observed on the "Wednesday"
    dates = pd.date_range(end=lcd, periods=7 * (cfg["energy_train_weeks"] + 2), freq="D")
    dates = dates[wday_u(dates) == wday_u(td)][-cfg["energy_train_weeks"]:]
    if cfg["drop_holidays_from_training"]:
        bad = is_special_day(dates) | is_special_day(dates - ref_lag * DAY) | is_special_day(dates - week_lag * DAY)
        dates = dates[~bad]

    y = V.reindex(dates).to_numpy()
    X = np.column_stack([V.reindex(dates - ref_lag * DAY), V.reindex(dates - week_lag * DAY)])
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    x_new = np.array([V.get(td - ref_lag * DAY, np.nan), V.get(td - week_lag * DAY, np.nan)])

    use = ~np.isnan(x_new)                              # guard: drop a regressor if its value is missing
    q, _ = ols_quantiles(X[ok][:, use], y[ok], x_new[use], spread=cfg["energy_spread"][col])

    # Course benchmark: empirical quantiles of the last 20 values of the same weekday and hour
    ####Benchmark detailed from the course slides:####
    hist = V[(V.index <= lcd) & (wday_u(V.index) == wday_u(td))].dropna()
    bench = np.quantile(hist.tail(20), QUANTILES) # empirical quantiles of the last 20 values (in this case 20 last fridays at 12)

    meta = dict(target=col, horizon=horizon, target_time=f"{td:%Y-%m-%d} {hour:02d}:00", obs=V.get(td, np.nan))
    return [dict(meta, model="model", **dict(zip(QCOLS, q))),
            dict(meta, model="benchmark", **dict(zip(QCOLS, bench)))]


# ================================================================== WEATHER ==
def forecast_weather(W, var, fd, horizon, cfg):
    s = W[var]
    t0 = s.loc[: fd.tz_localize("UTC") - H].last_valid_index()   # last DWD obs on Wednesday (Tue 23 UTC)
    T = (fd + DAY).tz_localize("UTC") + horizon * H                # target time in UTC
    k0 = int(np.ceil((T - t0) / H / 24))                           # days back to the last observed same clock hour
    lag = k0 * 24 * H                                              # e.g. Fri 12 UTC <- Tue 12 UTC (72 h)

    # Training rows: past target times at the same clock hour, observed before t0, same season
    Td = T - pd.to_timedelta(np.arange(k0, int(365.25 * cfg["weather_train_years"])) * 24, unit="h")
    dist = np.abs(Td.dayofyear - T.dayofyear)
    Td = Td[np.minimum(dist, 365 - dist) <= cfg["weather_season_days"]]

    y = s.reindex(Td).to_numpy()
    x = s.reindex(Td - lag).to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(x)

    clim = np.quantile(y[~np.isnan(y)], QUANTILES)                 # reference: seasonal climatology
    x_new = s.get(T - lag, np.nan)
    if np.isnan(x_new):                                            # guard: no lagged value -> climatology
        q = clim.copy()
    else:
        q, _ = ols_quantiles(x[ok, None], y[ok], x_new, spread=cfg["weather_spread"][var])
    if var == "wind":
        q, clim = np.clip(q, 0, None), np.clip(clim, 0, None)

    meta = dict(target=var, horizon=horizon, target_time=f"{T:%Y-%m-%d %H:%M} UTC", obs=s.get(T, np.nan))
    return [dict(meta, model="model", **dict(zip(QCOLS, q))),
            dict(meta, model="climatology", **dict(zip(QCOLS, clim)))]


# ===================================================================== ALL ==
def forecast_all(ED, W, cfg, fd):
    rows = []
    for h in HORIZONS:
        for col in ("demand", "res_demand"):
            rows += forecast_energy(ED, col, fd, h, cfg)
        for var in ("temperature", "wind"):
            rows += forecast_weather(W, var, fd, h, cfg)
    return pd.DataFrame(rows)
