"""
MODELS (v2) - still deliberately simple.

ENERGY (demand, res_demand): one multiple linear regression per target slot
    y(day, hour) = b0 + b1 * y(last complete day, same hour) + b2 * y(same slot one week earlier)
                   + c' * weather forecast for that hour (v2) + e
    trained on the last N observations of the same weekday and hour.
    Weather forecast = ECMWF run of the submission Wednesday, averaged over Germany: wind power proxy,
    solar radiation, temperature and heating degrees (residual load = load - wind - solar).
    "lagonly" = the same regression without weather (the v1 model), kept for comparison.

WEATHER (temperature, wind): cfg["weather_method"] chooses the model that is submitted ("model"):
    "enhanced" (v2b): OLS on the last two observed same-clock-hour values, variant chosen with
               cfg["weather_enhanced_variant"] (a string, or a dict per variable):
                 lag2          two lags, same-season training window (+/- weather_season_days)
                 full_lag2     two lags, full training window
                 fourier       two lags, full window + annual sin/cos
                 fourier_lag1  one lag,  full window + annual sin/cos
               Falls back to MOS, then to lagreg, if it cannot be fitted.
    "mos" (v2):  y(T) = a + b * ECMWF forecast for T issued on the Wednesday + e
               corrects the bias of the forecast at the station. Trained on the last N days + the same
               season in earlier years (forecast archive from Jan 2024). Wind on a square-root scale.
    "lagreg" = v1 regression on the last observed same-hour value (same-season training), kept as
               fallback (no forecast data) and for comparison.
    The models that are not submitted are still written as extra rows ("mos", "lagreg") for comparison.

UNCERTAINTY (all targets): q_tau = point forecast + spread * empirical tau-quantile of the residuals.
    v2 uses leave-one-out residuals (out-of-sample errors), so the spreads go back to 1.0.

Every forecast only uses information available on the submission Wednesday, so the same
functions are reused for the backtest (just call them with a past Wednesday). live=True is only
used for the real submission: it allows the latest forecast run to fill gaps in the Wednesday run.

COEFFICIENTS: pass a list as `coefs` and every regression appends its coefficient table
(beta, se, t, impact = beta * sd of the regressor) -> saved as output/coefficients_<date>.csv.
"""
import numpy as np
import pandas as pd

from config import HORIZONS, QCOLS, QUANTILES
from .data_nwp import nwp_at
from .utils import is_special_day, ols_quantiles, ols_table, wday_u

DAY, H = pd.Timedelta(days=1), pd.Timedelta(hours=1)

# horizon -> (days after the Wednesday, hour ending in German local time)
ENERGY_SLOTS = {36: (2, 12), 40: (2, 16), 44: (2, 20), 60: (3, 12), 64: (3, 16), 68: (3, 20)}
MIN_ROWS = 30       # minimum training rows for the regression with weather regressors
ENH_MIN_ROWS = 30   # minimum training rows for the enhanced weather regression
ENH_VARIANTS = ("lag2", "full_lag2", "fourier", "fourier_lag1", "full_lag1")

# weather_method -> models tried in order; the first one that works is submitted as "model"
WEATHER_ORDER = {"enhanced": ("enhanced", "mos", "lagreg"), "mos": ("mos", "lagreg"), "lagreg": ("lagreg",)}


def lead_day(fd, day, cfg):
    """NWP lead day = calendar days between the Wednesday and the target day (Fri 2, Sat 3)."""
    return int(min(max((pd.Timestamp(day).normalize() - fd).days, 1), cfg["nwp_max_lead_day"]))


def slot_utc(dates, hour):
    """End of the local-time hour (hour-1, hour] in UTC (radiation is the mean of the preceding hour)."""
    local = pd.DatetimeIndex(dates) + pd.Timedelta(hours=hour)
    return local.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


# =================================================================== ENERGY ==
def forecast_energy(ED, col, fd, horizon, cfg, Ne=None, live=False, coefs=None):
    day_offset, hour = ENERGY_SLOTS[horizon]
    V = ED.wide[col][hour]                              # one value per day: this target at this hour
    td = fd + day_offset * DAY                          # target day (Friday or Saturday)
    lcd = ED.last_complete_day(col, fd - DAY)           # latest complete day known on Wednesday (Tuesday)
    ref_lag = (td - lcd).days                           # Fri - Tue = 3, Sat - Tue = 4
    week_lag = 7 * int(np.ceil(ref_lag / 7))            # same slot one week earlier (7)

    # Training rows: same weekday as the target, already observed on the "Wednesday"
    dates = pd.date_range(end=lcd, periods=7 * (cfg["energy_train_weeks"] + 2), freq="D")
    dates = dates[wday_u(dates) == wday_u(td)][-cfg["energy_train_weeks"]:]
    if cfg["drop_holidays_from_training"]:
        bad = is_special_day(dates) | is_special_day(dates - ref_lag * DAY) | is_special_day(dates - week_lag * DAY)
        dates = dates[~bad]

    y = V.reindex(dates).to_numpy()
    X = np.column_stack([V.reindex(dates - ref_lag * DAY), V.reindex(dates - week_lag * DAY)])
    x_new = np.array([V.get(td - ref_lag * DAY, np.nan), V.get(td - week_lag * DAY, np.nan)])

    tables = {}                                         # coefficient tables, keyed by specification

    def fit(X, x_new, names, spec, min_rows=3):
        use = ~np.isnan(x_new)                          # guard: drop a regressor if its value is missing
        ok = ~np.isnan(y) & ~np.isnan(X[:, use]).any(axis=1)
        if ok.sum() < min_rows:
            return None
        if coefs is not None:
            tables[spec] = ols_table(X[ok][:, use], y[ok], [n for n, u in zip(names, use) if u])
        return ols_quantiles(X[ok][:, use], y[ok], x_new[use], spread=cfg["energy_spread"][col],
                             loo=cfg["residuals"] == "loo")[0]

    lag_names = ["y_last_day", "y_week_before"]
    q_lag = fit(X, x_new, lag_names, "lags")

    # v2: add the weather forecast for the target hour (same lead day in training and in real time)
    q_nwp = None
    feats = cfg["energy_nwp_features"][col] if cfg["energy_use_nwp"] and Ne is not None else []
    if feats:
        k = lead_day(fd, td, cfg)
        Z = np.column_stack([nwp_at(Ne, f, k, slot_utc(dates, hour)) for f in feats])
        z_new = np.array([nwp_at(Ne, f, k, slot_utc([td], hour), live=live)[0] for f in feats])
        if not np.isnan(z_new).all():
            q_nwp = fit(np.column_stack([X, Z]), np.concatenate([x_new, z_new]), lag_names + [f"nwp_{f}" for f in feats],
                        "lags+nwp", min_rows=MIN_ROWS)

    # Course benchmark: empirical quantiles of the last 20 values of the same weekday and hour
    hist = V[(V.index <= lcd) & (wday_u(V.index) == wday_u(td))].dropna()
    bench = np.quantile(hist.tail(20), QUANTILES)

    meta = dict(target=col, horizon=horizon, target_time=f"{td:%Y-%m-%d} {hour:02d}:00", obs=V.get(td, np.nan))
    rows = [dict(meta, model="model", **dict(zip(QCOLS, q_lag if q_nwp is None else q_nwp))),
            dict(meta, model="benchmark", **dict(zip(QCOLS, bench)))]
    if q_nwp is not None:                               # v1 model, only for comparison
        rows.append(dict(meta, model="lagonly", **dict(zip(QCOLS, q_lag))))
    if coefs is not None:
        names = {"lags": "model" if q_nwp is None else "lagonly", "lags+nwp": "model"}
        coefs += [dict(origin=fd, target=col, horizon=horizon, model=names[k], **r)
                  for k, tab in tables.items() if k == "lags" or q_nwp is not None for r in tab]
    return rows


# ================================================================== WEATHER ==
def forecast_mos(W, S, var, fd, T, t0, cfg, live=False, coefs=None, horizon=None):
    """MOS: regression of the observation on the ECMWF forecast with the same lead day. None if not possible."""
    k = lead_day(fd, T.tz_localize(None), cfg)
    sqrt = var == "wind" and cfg["wind_sqrt"]
    f = (lambda v: np.sqrt(np.clip(v, 0, None))) if sqrt else (lambda v: np.asarray(v, float))

    # Training rows: same clock hour, observed before t0; last N days + same season in earlier years
    n = int((T - pd.Timestamp(cfg["nwp_start_date"], tz="UTC")) / DAY)
    Td = T - pd.to_timedelta(np.arange(1, max(n, 2)) * 24, unit="h")
    Td = Td[Td <= t0]
    dist = np.abs(Td.dayofyear - T.dayofyear)
    Td = Td[(Td > t0 - cfg["mos_window_days"] * DAY) | (np.minimum(dist, 365 - dist) <= cfg["mos_season_days"])]

    x, y = f(nwp_at(S, var, k, Td)), f(W[var].reindex(Td).to_numpy())
    x_new = f(nwp_at(S, var, k, [T], live=live))[0]
    ok = ~np.isnan(x) & ~np.isnan(y)
    if np.isnan(x_new) or ok.sum() < cfg["mos_min_rows"]:
        return None

    q, _ = ols_quantiles(x[ok, None], y[ok], x_new, spread=cfg["weather_spread"][var], loo=cfg["residuals"] == "loo")
    if coefs is not None:                               # wind: coefficients on the square-root scale
        coefs += [dict(origin=fd, target=var, horizon=horizon, model="model", **r)
                  for r in ols_table(x[ok, None], y[ok], ["ecmwf_sqrt" if sqrt else "ecmwf"])]
    return np.clip(q, 0, None) ** 2 if sqrt else q


def enhanced_variant(var, cfg):
    v = cfg.get("weather_enhanced_variant", "lag2")
    v = v.get(var, "lag2") if isinstance(v, dict) else v
    if v not in ENH_VARIANTS:
        raise ValueError(f"Unknown weather_enhanced_variant for {var}: {v} (use one of {ENH_VARIANTS})")
    return v


def forecast_weather_enhanced(W, var, fd, T, t0, cfg, coefs=None, horizon=None):
    """Enhanced lag regression (see module docstring). Returns the quantiles, or None if not possible."""
    variant = enhanced_variant(var, cfg)
    s = W[var]
    k0 = int(np.ceil((T - t0) / H / 24))                # first same clock hour observed before t0
    lag1, lag2 = k0 * 24 * H, (k0 + 1) * 24 * H         # e.g. Fri 12 UTC <- Tue 12 UTC and Mon 12 UTC

    # Training rows: past target times at the same clock hour, never after t0
    Td = T - pd.to_timedelta(np.arange(k0, int(365.25 * cfg["weather_train_years"])) * 24, unit="h")
    if variant == "lag2":                               # same season only (as the v1 model)
        dist = np.abs(Td.dayofyear - T.dayofyear)
        Td = Td[np.minimum(dist, 365 - dist) <= cfg["weather_season_days"]]

    y = s.reindex(Td).to_numpy(float)
    cols, x_new, names = [s.reindex(Td - lag1).to_numpy(float)], [s.get(T - lag1, np.nan)], ["y_lag1"]
    if variant not in ("fourier_lag1", "full_lag1"):
        cols.append(s.reindex(Td - lag2).to_numpy(float))
        x_new.append(s.get(T - lag2, np.nan))
        names.append("y_lag2")
    if variant.startswith("fourier"):                   # explicit annual seasonality
        w, doy = 2 * np.pi / 365.25, Td.dayofyear.to_numpy()
        cols += [np.sin(w * doy), np.cos(w * doy)]
        x_new += [np.sin(w * T.dayofyear), np.cos(w * T.dayofyear)]
        names += ["sin_doy", "cos_doy"]

    X, x_new = np.column_stack(cols), np.array(x_new, float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if not np.isfinite(x_new).all() or ok.sum() < ENH_MIN_ROWS:
        return None

    q, _ = ols_quantiles(X[ok], y[ok], x_new, spread=cfg["weather_spread"][var], loo=cfg["residuals"] == "loo")
    if coefs is not None:
        coefs += [dict(origin=fd, target=var, horizon=horizon, model="model", variant=variant, **r)
                  for r in ols_table(X[ok], y[ok], names)]
    return np.clip(q, 0, None) if var == "wind" else q


def forecast_weather(W, var, fd, horizon, cfg, S=None, live=False, coefs=None):
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
        q, _ = ols_quantiles(x[ok, None], y[ok], x_new, spread=cfg["weather_spread"][var],
                             loo=cfg["residuals"] == "loo")
    if var == "wind":
        q, clim = np.clip(q, 0, None), np.clip(clim, 0, None)

    # Candidate models; cfg["weather_method"] decides which one is submitted (with fallbacks)
    method = cfg["weather_method"]
    cand, tabs = {"lagreg": q}, {}
    if coefs is not None and not np.isnan(x_new) and ok.sum() > 2:
        tabs["lagreg"] = [dict(origin=fd, target=var, horizon=horizon, **r)
                          for r in ols_table(x[ok, None], y[ok], ["y_lag"])]
    if method in ("mos", "enhanced") and S is not None:            # v2: MOS on the ECMWF forecast
        tabs["mos"] = [] if coefs is not None else None
        cand["mos"] = forecast_mos(W, S, var, fd, T, t0, cfg, live, tabs["mos"], horizon)
    if method == "enhanced":                                       # v2b: enhanced lag regression
        tabs["enhanced"] = [] if coefs is not None else None
        cand["enhanced"] = forecast_weather_enhanced(W, var, fd, T, t0, cfg, tabs["enhanced"], horizon)
    final = next(m for m in WEATHER_ORDER.get(method, ("lagreg",)) if cand.get(m) is not None)

    if coefs is not None:
        for name, tab in tabs.items():
            for r in tab or []:
                r["model"] = "model" if name == final else name
            coefs += tab or []

    meta = dict(target=var, horizon=horizon, target_time=f"{T:%Y-%m-%d %H:%M} UTC", obs=s.get(T, np.nan))
    rows = [dict(meta, model="model", **dict(zip(QCOLS, cand[final]))),
            dict(meta, model="climatology", **dict(zip(QCOLS, clim)))]
    rows += [dict(meta, model=m, **dict(zip(QCOLS, cand[m])))      # models not submitted, only for comparison
             for m in ("mos", "lagreg") if m != final and cand.get(m) is not None]
    return rows


# ===================================================================== ALL ==
def forecast_all(ED, W, cfg, fd, NWP=None, live=False, energy=True, weather=True, coefs=None):
    NWP = NWP or {}
    rows = []
    for h in HORIZONS:
        if energy:
            for col in ("demand", "res_demand"):
                rows += forecast_energy(ED, col, fd, h, cfg, NWP.get("energy"), live, coefs)
        if weather:
            for var in ("temperature", "wind"):
                rows += forecast_weather(W, var, fd, h, cfg, NWP.get("station"), live, coefs)
    return pd.DataFrame(rows)
