"""Backtest: re-run the exact same forecast for each of the last N Wednesdays and score it."""
import numpy as np
import pandas as pd

from config import QCOLS
from .models import forecast_all
from .utils import mean_qs


def run_backtest(ED, W, cfg, fd):
    origins = [fd - pd.Timedelta(weeks=k) for k in range(1, cfg["backtest_weeks"] + 1)]
    B = pd.concat([forecast_all(ED, W, cfg, o).assign(origin=o) for o in origins], ignore_index=True)
    B = B.dropna(subset=["obs"]).copy()
    B["qs"] = mean_qs(B[QCOLS].to_numpy(), B["obs"].to_numpy())
    B["in50"] = (B["obs"] >= B["q0.25"]) & (B["obs"] <= B["q0.75"])
    B["in95"] = (B["obs"] >= B["q0.025"]) & (B["obs"] <= B["q0.975"])
    return B


def summarise(B, by=("target", "model")):
    g = B.assign(bias=B["obs"] - B["q0.5"]).groupby(list(by))
    out = pd.DataFrame({"mean_QS": g["qs"].mean(), "cov50": g["in50"].mean(), "cov95": g["in95"].mean(),
                        "bias_median": g["bias"].mean(), "n": g.size()})
    return out.round(3).reset_index()
