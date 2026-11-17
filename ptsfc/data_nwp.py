"""
NWP data (v2): ECMWF IFS 0.25 deg forecasts from the Open-Meteo Previous Runs API.

To train a model on weather FORECASTS we need historical forecasts issued at the same lead time
as the real ones. The Previous Runs API stores "<var>_previous_dayN" = value forecast N days
before the valid day (archive from Jan 2024); "<var>" alone is the latest run.
We standardise the columns to "<name>_d<N>" (N = 0 latest run, 1..3 lead days).

Lead day = calendar days between the submission Wednesday and the target day (UTC):
Friday -> d2, Saturday -> d3. The same column is used in training and in real time.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

from .utils import msg


def _get(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and js.get("error"):
                raise RuntimeError(js.get("reason"))
            return js
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Open-Meteo request failed ({last})")


def _to_frame(js):
    h = js["hourly"]
    idx = pd.DatetimeIndex(pd.to_datetime(h["time"], format="%Y-%m-%dT%H:%M")).tz_localize("UTC")
    return pd.DataFrame({k: np.array([np.nan if v is None else v for v in vals], float)
                         for k, vals in h.items() if k != "time"}, index=idx)


def fetch_point(cfg, name, lat, lon, var, end):
    """Hourly series of one variable at one location, all lead days; cached in data/ and refreshed incrementally."""
    L = cfg["nwp_max_lead_day"]
    cols = [var] + [f"{var}_previous_day{k}" for k in range(1, L + 1)]
    path = os.path.join(cfg["dir_data"], f"nwp_{cfg['nwp_model']}_{name}_{var}.pkl")
    cache = pd.read_pickle(path) if os.path.exists(path) else None

    if not cfg["offline"]:
        start = pd.Timestamp(cfg["nwp_start_date"])
        if cache is not None:   # re-download the recent past + the future (forecasts still change)
            last = min(cache.dropna(how="all").index.max().tz_localize(None), pd.Timestamp.today()).normalize()
            start = max(start, last - pd.Timedelta(days=cfg["nwp_refresh_days"]))
        parts, s = [], start
        while s <= end:
            e = min(end, s + pd.Timedelta(days=cfg["nwp_chunk_days"] - 1))
            p = dict(latitude=f"{lat:.4f}", longitude=f"{lon:.4f}", hourly=",".join(cols), models=cfg["nwp_model"],
                     wind_speed_unit="ms", timezone="GMT", start_date=f"{s:%Y-%m-%d}", end_date=f"{e:%Y-%m-%d}")
            try:
                parts.append(_to_frame(_get(cfg["nwp_url"], p)))
            except Exception as ex:  # noqa: BLE001
                msg("WARNING: NWP %s %s..%s failed (%s)", name, s.date(), e.date(), ex)
            s = e + pd.Timedelta(days=1)
        if parts:
            new = pd.concat(parts)
            new = new[~new.index.duplicated(keep="last")]
            cache = new if cache is None else new.combine_first(cache).sort_index()   # new values win
            cache.to_pickle(path)

    if cache is None:
        raise RuntimeError(f"no NWP data for {name} ({var})")
    out = cache.reindex(columns=cols)
    out.columns = [f"x_d{k}" for k in range(L + 1)]
    return out


def wind_power(v, cut_in=3.0, rated=12.0, cut_out=25.0):
    """Normalised power (0..1) of a generic turbine: cubic between cut-in and rated speed, 0 above cut-out."""
    v = np.asarray(v, float)
    p = np.where(v >= cut_out, 0.0, np.clip((v**3 - cut_in**3) / (rated**3 - cut_in**3), 0, 1))
    return np.where(np.isnan(v), np.nan, p)


def load_station_nwp(cfg, end):
    """Frankfurt Airport: columns temperature_d0..3 and wind_d0..3."""
    S = pd.concat([fetch_point(cfg, "station", cfg["station_lat"], cfg["station_lon"], v, end)
                   .rename(columns=lambda c, n=n: c.replace("x", n, 1))
                   for n, v in (("temperature", "temperature_2m"), ("wind", "wind_speed_10m"))], axis=1)
    msg("NWP station: %s to %s", S.index.min(), S.index.max())
    return S


def load_energy_nwp(cfg, end):
    """
    Germany-wide regressors per lead day (columns <name>_d0..3), averaged over the locations in config:
    wind_power (power-curve proxy of 100 m wind, 0..1), solar (shortwave radiation, W/m2),
    temp (2 m temperature, deg C) and hdd (heating degrees).
    """
    pts, out = cfg["energy_nwp_points"], {}

    def mean_over(group, var, fun=None):
        parts = []
        for name, lat, lon in pts[group]:
            try:
                d = fetch_point(cfg, name, lat, lon, var, end)
                parts.append(d.apply(lambda c: pd.Series(fun(c), index=c.index)) if fun else d)
            except Exception as e:  # noqa: BLE001
                msg("WARNING: NWP point %s skipped (%s)", name, e)
        return pd.concat(parts).groupby(level=0).mean() if parts else None

    out["wind_power"] = mean_over("wind", "wind_speed_100m", lambda c: wind_power(c, **cfg["wind_power_curve"]))
    out["solar"] = mean_over("solar", "shortwave_radiation")
    out["temp"] = mean_over("load", "temperature_2m")
    if out["temp"] is not None:
        out["hdd"] = (cfg["hdd_base"] - out["temp"]).clip(lower=0)
    out = {k: v for k, v in out.items() if v is not None}
    if not out:
        raise RuntimeError("no energy NWP features")
    E = pd.concat([v.rename(columns=lambda c, k=k: c.replace("x", k, 1)) for k, v in out.items()], axis=1)
    msg("NWP energy regressors: %s | %s to %s", sorted(out), E.index.min(), E.index.max())
    return E


def load_nwp(cfg, fd):
    """dict(station=..., energy=...); an entry is None if its data are unavailable (models then fall back)."""
    NWP = dict(station=None, energy=None)
    if not cfg["nwp_enabled"]:
        return NWP
    end = max(pd.Timestamp.today().normalize(), fd) + pd.Timedelta(days=5)
    for key, fun in (("station", load_station_nwp), ("energy", load_energy_nwp)):
        try:
            NWP[key] = fun(cfg, end)
        except Exception as e:  # noqa: BLE001
            msg("WARNING: NWP %s unavailable (%s) -> fallback model", key, e)
    return NWP


def nwp_at(df, name, lead_day, times, live=False):
    """
    Values of <name>_d<lead_day> at the given UTC times.
    live=True fills gaps with the latest run (_d0). Only for the real-time forecast: in training,
    _d0 is a short-lead forecast and would make the backtest look better than reality.
    """
    times = pd.DatetimeIndex(times)
    col = f"{name}_d{lead_day}"
    if df is None or col not in df.columns:
        return np.full(len(times), np.nan)
    v = np.array(df[col].reindex(times).to_numpy(float))
    if live and np.isnan(v).any():
        v = np.where(np.isnan(v), df[f"{name}_d0"].reindex(times).to_numpy(float), v)
    return v