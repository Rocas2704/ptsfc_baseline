"""
Energy data: SMARD (German Federal Network Agency)
- API: https://www.smard.de/app/chart_data/{filter}/DE/...
- or CSV exported from smard.de / Ilias

Output: hourly table containing date (German local time), hour_end (1..24),
demand, and res_demand in GWh.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

from .utils import msg, wday_u


SMARD_BASE = "https://www.smard.de/app/chart_data"


def get_json(url, tries=3):
    last = None

    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()

        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))

    raise RuntimeError(f"Could not download: {url} ({last})")


def smard_parse_series(js):
    s = js.get("series") or []

    ts = [float(p[0]) for p in s]
    val = [np.nan if p[1] is None else float(p[1]) for p in s]

    return pd.DataFrame({
        "ts": ts,
        "value": val,
    })


def fetch_smard_filter(filt, cfg, refresh_last_n=3):
    cache_file = os.path.join(
        cfg["dir_data"],
        f"smard_{filt}_{cfg['smard_resolution']}.pkl",
    )

    cache = (
        pd.read_pickle(cache_file)
        if os.path.exists(cache_file)
        else pd.DataFrame({
            "chunk": pd.Series(dtype=float),
            "ts": pd.Series(dtype=float),
            "value": pd.Series(dtype=float),
        })
    )

    if cfg["offline"]:
        if cache.empty:
            raise RuntimeError(
                f"Offline mode enabled, but no cache is available: {cache_file}"
            )

        return cache[["ts", "value"]]

    idx = get_json(
        f"{SMARD_BASE}/{filt}/DE/index_{cfg['smard_resolution']}.json"
    )["timestamps"]

    start_ms = (
        pd.Timestamp(
            cfg["energy_start_date"],
            tz="Europe/Berlin",
        ).timestamp()
        * 1000
        - 8 * 86400 * 1000
    )

    idx = sorted(t for t in idx if t >= start_ms)

    have = set(cache["chunk"].unique())
    todo = sorted(
        set(t for t in idx if t not in have)
        | set(idx[-refresh_last_n:])
    )

    msg(
        "SMARD filter %s: %d weekly blocks to download",
        filt,
        len(todo),
    )

    new = []

    for ch in todo:
        url = (
            f"{SMARD_BASE}/{filt}/DE/"
            f"{filt}_DE_{cfg['smard_resolution']}_{int(ch)}.json"
        )

        part = smard_parse_series(get_json(url))
        part["chunk"] = float(ch)
        new.append(part)

    if new:
        cache = pd.concat(
            [
                cache[~cache["chunk"].isin(todo)],
                *new,
            ],
            ignore_index=True,
        )

    cache = cache.sort_values("ts").reset_index(drop=True)
    cache.to_pickle(cache_file)

    return cache[["ts", "value"]]


def _local_columns(t_utc):
    loc = t_utc.dt.tz_convert("Europe/Berlin")

    return (
        loc.dt.tz_localize(None).dt.normalize(),
        loc.dt.hour + 1,
    )


def smard_to_hourly(df):
    """
    Aggregate data to hourly resolution (in case quarter-hourly data is provided)
    and convert timestamps to German local time.

    MWh -> GWh.
    """
    df = df.dropna(subset=["ts"]).copy()

    step = (
        np.median(np.diff(np.sort(df["ts"].unique())))
        if df["ts"].nunique() > 1
        else 3.6e6
    )

    expected = max(1, round(3.6e6 / step))

    df["hts"] = np.floor(df["ts"] / 3.6e6) * 3.6e6
    g = df.groupby("hts")["value"]

    h = pd.DataFrame({
        "n": g.count(),
        "value": g.sum(min_count=1),
    }).reset_index()

    h.loc[h["n"] != expected, "value"] = np.nan

    t_utc = pd.to_datetime(
        h["hts"],
        unit="ms",
        utc=True,
    )

    date, hour_end = _local_columns(t_utc)

    return pd.DataFrame({
        "time_utc": t_utc,
        "date": date,
        "hour_end": hour_end.astype(int),
        "value": h["value"] / 1000,
    })


def read_smard_csv(path, cols):
    raw = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        encoding="utf-8-sig",
    )

    def num(x):
        x = (
            x.astype(str)
            .str.strip()
            .str.replace(r"^-$", "", regex=True)
        )

        x = (
            x.str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        return pd.to_numeric(x, errors="coerce")

    start = raw.iloc[:, 0].astype(str)

    d = pd.DataFrame({
        "date": pd.to_datetime(
            start.str[:10],
            format="%d.%m.%Y",
        ),
        "hour_start": start.str[11:13].astype(int),
        "demand": num(raw.iloc[:, cols["demand"] - 1]),
        "res_demand": num(raw.iloc[:, cols["res_demand"] - 1]),
    })

    n_per_hour = (
        d.groupby(["date", "hour_start"])
        .size()
        .median()
    )

    h = (
        d.groupby(["date", "hour_start"])
        .agg(
            n=("demand", "size"),
            demand=("demand", lambda x: x.sum(min_count=len(x))),
            res_demand=("res_demand", lambda x: x.sum(min_count=len(x))),
        )
    )

    h = h[h["n"] == n_per_hour].reset_index()

    h["hour_end"] = h["hour_start"] + 1
    h[["demand", "res_demand"]] = h[["demand", "res_demand"]] / 1000

    local = h["date"] + pd.to_timedelta(
        h["hour_start"],
        unit="h",
    )

    h["time_utc"] = (
        local.dt.tz_localize(
            "Europe/Berlin",
            ambiguous="NaT",
            nonexistent="NaT",
        )
        .dt.tz_convert("UTC")
    )

    return h.dropna(subset=["time_utc"])[[
        "time_utc",
        "date",
        "hour_end",
        "demand",
        "res_demand",
    ]]


class EnergyData:
    """
    Hourly table plus wide-format views (date x hour)
    for efficient lag lookups.
    """

    def __init__(self, E):
        self.E = E
        self.wide, self.daily = {}, {}

        for col in ("demand", "res_demand"):
            w = E.pivot(
                index="date",
                columns="hour_end",
                values=col,
            )

            w = w.reindex(
                pd.date_range(
                    w.index.min(),
                    w.index.max(),
                    freq="D",
                )
            )

            self.wide[col] = w
            self.daily[col] = w.mean(axis=1).where(
                w.notna().sum(axis=1) >= 23
            )

    def last_complete_day(self, col, max_day):
        n = self.wide[col].notna().sum(axis=1)
        ok = n[(n >= 23) & (n.index <= max_day)]

        return ok.index.max()


def load_energy_data(cfg):
    if cfg["energy_source"] == "csv":
        msg(
            "Reading energy data from CSV: %s",
            cfg["energy_csv_path"],
        )

        E = read_smard_csv(
            cfg["energy_csv_path"],
            cfg["energy_csv_cols"],
        )

    else:
        parts = []

        for nm, filt in cfg["smard_filters"].items():
            h = smard_to_hourly(
                fetch_smard_filter(filt, cfg)
            ).rename(columns={"value": nm})

            parts.append(h)

        E = parts[0]

        for p in parts[1:]:
            E = E.merge(
                p,
                on=["time_utc", "date", "hour_end"],
                how="outer",
            )

    E = E[
        E["date"] >= pd.Timestamp(cfg["energy_start_date"])
    ]

    # Daylight saving time transition: remove duplicated hour
    E = E.drop_duplicates(subset=["date", "hour_end"])
    E = E.sort_values("time_utc").reset_index(drop=True)

    E["wday"] = wday_u(E["date"])

    fmt = lambda c: (  # noqa: E731
        E.loc[E[c].notna(), "time_utc"]
        .max()
        .tz_convert("Europe/Berlin")
    )

    msg(
        "Energy data: %s to %s | "
        "latest demand observation: %s | "
        "latest res_demand observation: %s",
        E["date"].min().date(),
        E["date"].max().date(),
        fmt("demand"),
        fmt("res_demand"),
    )

    return EnergyData(E)