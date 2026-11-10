"""
Backtest: re-run the exact same forecast for past origins and score it.

Energy uses the last N Wednesdays. Weather has no weekday effect, so with backtest_weather_daily
every day is used as an origin (~7x more cases -> much tighter confidence intervals).
compare() then scores candidate vs. reference on exactly the same origins (paired comparison).
"""
import numpy as np
import pandas as pd

from config import QCOLS
from .models import forecast_all
from .utils import block_bootstrap_ci, mean_qs


def run_backtest(ED, W, cfg, fd, NWP=None, coefs=None):
    """coefs: optional list that collects the coefficient tables of every origin (see models.py)."""
    daily = cfg["backtest_weather_daily"]
    weeks = [fd - pd.Timedelta(weeks=k) for k in range(1, cfg["backtest_weeks"] + 1)]
    B = [forecast_all(ED, W, cfg, o, NWP, weather=not daily, coefs=coefs).assign(origin=o) for o in weeks]
    if daily:
        days = pd.date_range(fd - pd.Timedelta(weeks=cfg["backtest_weeks"]), fd - pd.Timedelta(days=1))
        B += [forecast_all(ED, W, cfg, o, NWP, energy=False, coefs=coefs).assign(origin=o) for o in days]
    B = pd.concat(B, ignore_index=True).dropna(subset=["obs"]).copy()
    B["qs"] = mean_qs(B[QCOLS].to_numpy(), B["obs"].to_numpy())
    B["in50"] = (B["obs"] >= B["q0.25"]) & (B["obs"] <= B["q0.75"])
    B["in95"] = (B["obs"] >= B["q0.025"]) & (B["obs"] <= B["q0.975"])
    return B


def summarise(B, by=("target", "model")):
    g = B.assign(bias=B["obs"] - B["q0.5"]).groupby(list(by))
    out = pd.DataFrame({"mean_QS": g["qs"].mean(), "cov50": g["in50"].mean(), "cov95": g["in95"].mean(),
                        "bias_median": g["bias"].mean(), "n": g.size()})
    return out.round(3).reset_index()


def compare(B, cfg):
    """
    Paired comparison candidate vs. reference on the same (origin, horizon) cases.
    Scores are averaged over the horizons of each origin first (horizons of one origin are not independent).
    gain = mean(QS_ref - QS_cand) > 0 favours the candidate; CI = circular moving-block bootstrap.
    The CI is exploratory: the same period was used to build the models.
    """
    out = []
    for tg, ref, cand in cfg["comparisons"]:
        P = B[(B.target == tg) & B.model.isin([ref, cand])].pivot_table(
            index=["origin", "horizon"], columns="model", values="qs")
        if ref not in P or cand not in P or P.dropna().empty:
            continue
        P = P.dropna().groupby(level="origin").mean().sort_index()
        d = (P[ref] - P[cand]).to_numpy()
        step = max(1, int(np.median(np.diff(P.index.values)) / np.timedelta64(1, "D"))) if len(P) > 1 else 7
        lo, hi = block_bootstrap_ci(d, round(cfg["bootstrap_block_days"] / step), cfg["bootstrap_n"])
        out.append(dict(target=tg, reference=ref, candidate=cand, n_origins=len(d), QS_ref=P[ref].mean(),
                        QS_cand=P[cand].mean(), gain=d.mean(), rel_gain=d.mean() / P[ref].mean(),
                        win_share=np.mean(d > 0), ci_lo=lo, ci_hi=hi,
                        verdict="candidate better" if lo > 0 else "reference better" if hi < 0 else "no clear difference"))
    return pd.DataFrame(out).round(4) if out else None