"""
Weather data
- DWD observations from station 1420 (Frankfurt Airport), hourly, UTC
"""
import io
import os
import re
import zipfile

import pandas as pd
import requests

from .utils import msg


DWD_BASE = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly"
)

DWD_VARS = {
    "temperature": dict(
        dir="air_temperature",
        prefix="TU",
        col="TT_TU",
    ),
    "wind": dict(
        dir="wind",
        prefix="FF",
        col="F",
    ),
}


def dwd_read_produkt(buf, col):
    df = pd.read_csv(
        buf,
        sep=";",
        dtype=str,
        skipinitialspace=True,
    )

    df.columns = [c.strip() for c in df.columns]

    val = pd.to_numeric(
        df[col].str.strip(),
        errors="coerce",
    )
    val = val.where(val > -998)

    t = pd.to_datetime(
        df["MESS_DATUM"].str.strip(),
        format="%Y%m%d%H",
        utc=True,
    )

    return pd.DataFrame({
        "time": t,
        "value": val.values,
    })


def dwd_zip_to_df(url, col):
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        prod = [
            n
            for n in z.namelist()
            if re.match(r"produkt_.*\.txt$", os.path.basename(n))
        ]

        if not prod:
            raise RuntimeError(f"No produkt_*.txt file found in {url}")

        with z.open(prod[0]) as f:
            return dwd_read_produkt(
                io.TextIOWrapper(f, encoding="latin-1"),
                col,
            )


def dwd_find_hist_url(var, station):
    info = DWD_VARS[var]
    dir_url = f"{DWD_BASE}/{info['dir']}/historical/"

    html = requests.get(dir_url, timeout=60).text

    hits = sorted(
        set(
            re.findall(
                rf"stundenwerte_{info['prefix']}_"
                rf"{station:05d}_\d+_\d+_hist\.zip",
                html,
            )
        )
    )

    if not hits:
        raise RuntimeError(f"No historical ZIP archive found for {var}")

    return dir_url + hits[-1]


def fetch_dwd_var(var, cfg):
    info = DWD_VARS[var]
    st = cfg["dwd_station_id"]

    hist_cache = os.path.join(
        cfg["dir_data"],
        f"dwd_hist_{info['prefix']}_{st:05d}.pkl",
    )

    if os.path.exists(hist_cache):
        hist = pd.read_pickle(hist_cache)

    else:
        url = dwd_find_hist_url(var, st)

        msg(
            "DWD: downloading historical data %s",
            os.path.basename(url),
        )

        hist = dwd_zip_to_df(url, info["col"])
        hist.to_pickle(hist_cache)

    rec_url = (
        f"{DWD_BASE}/{info['dir']}/recent/"
        f"stundenwerte_{info['prefix']}_{st:05d}_akt.zip"
    )

    msg(
        "DWD: downloading recent data %s",
        os.path.basename(rec_url),
    )

    rec = dwd_zip_to_df(rec_url, info["col"])

    out = pd.concat([
        hist[~hist["time"].isin(rec["time"])],
        rec,
    ]).sort_values("time")

    return out.rename(columns={"value": var})


def load_weather_data(cfg):
    """
    Return a DataFrame with a complete hourly UTC index
    and columns for temperature and wind.
    """
    cache_file = os.path.join(
        cfg["dir_data"],
        f"dwd_{cfg['dwd_station_id']:05d}_weather.pkl",
    )

    if cfg["offline"]:
        if not os.path.exists(cache_file):
            raise RuntimeError(
                f"Offline mode enabled, but no cache is available: {cache_file}"
            )

        W = pd.read_pickle(cache_file)

    else:
        W = fetch_dwd_var("temperature", cfg).merge(
            fetch_dwd_var("wind", cfg),
            on="time",
            how="outer",
        )

        W.to_pickle(cache_file)

    W = (
        W.drop_duplicates("time")
        .set_index("time")
        .sort_index()
    )

    W = W.reindex(
        pd.date_range(
            W.index.min(),
            W.index.max(),
            freq="h",
            tz="UTC",
        )
    )[["temperature", "wind"]]

    W.index.name = "time"

    msg(
        "Weather: %s to %s | "
        "latest T2M observation: %s | "
        "latest wind observation: %s",
        W.index.min(),
        W.index.max(),
        W["temperature"].last_valid_index(),
        W["wind"].last_valid_index(),
    )

    return W
