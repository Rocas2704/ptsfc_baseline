"""Plots: a few preprocessing checks, calibration of the backtest, and the current forecast."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from config import QCOLS, QUANTILES, TARGETS  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
LAB = {"demand": "Load [GWh]", "res_demand": "Residual load [GWh]",
       "temperature": "2 m temperature [°C]", "wind": "10 m wind speed [m/s]"}
COL = {"model": "tab:purple", "benchmark": "tab:red", "climatology": "tab:green"}


# ------------------------------------------------------------ Preprocessing --
def plots_preprocessing(ED, W):
    E, P = ED.E, {}

    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    daily = E.groupby("date")[["demand", "res_demand"]].mean()
    we = daily.index.dayofweek >= 5
    for ax, c in zip(axs, ["demand", "res_demand"]):
        ax.scatter(daily.index[~we], daily[c][~we], s=2, label="Mon-Fri")
        ax.scatter(daily.index[we], daily[c][we], s=2, label="Sat/Sun")
        ax.set_title(LAB[c])
    axs[0].legend(markerscale=4)
    fig.suptitle("Energy: daily mean (SMARD)")
    P["pre_01_energy_series"] = fig

    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    sl = E[E.wday.isin([5, 6]) & E.hour_end.isin([12, 16, 20]) & (E.date >= E.date.max() - pd.Timedelta(days=730))]
    for ax, c in zip(axs, ["demand", "res_demand"]):
        for (wd, he), g in sl.groupby(["wday", "hour_end"]):
            ax.plot(g.date, g[c], lw=0.6, label=f"{'Fri' if wd == 5 else 'Sat'} {he}h")
        ax.set_title(LAB[c])
    axs[0].legend(ncol=6, fontsize=7)
    fig.suptitle("The 6 target slots (last 2 years)")
    P["pre_02_energy_target_slots"] = fig

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fri = ED.wide["demand"].index[ED.wide["demand"].index.dayofweek == 4]
    fri = fri[fri >= fri.max() - pd.Timedelta(days=730)]
    for i, c in enumerate(["demand", "res_demand"]):
        V = ED.wide[c][12]
        for j, (lag, name) in enumerate([(3, "Tuesday, same hour"), (7, "previous Friday, same hour")]):
            ax = axs[i, j]
            ax.scatter(V.reindex(fri - pd.Timedelta(days=lag)), V.reindex(fri), s=5, alpha=0.5)
            ax.set_xlabel(name); ax.set_ylabel(f"{c}, Friday 12h")
    fig.suptitle("Energy regressors: Friday 12h vs. its two lagged values")
    P["pre_03_energy_regressors"] = fig

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    recent = W[W.index >= W.index.max() - pd.Timedelta(days=90)]
    for j, v in enumerate(["temperature", "wind"]):
        axs[0, j].plot(recent.index, recent[v], lw=0.5); axs[0, j].set_title(f"{LAB[v]} - last 90 days")
        s = W[v][W.index >= W.index.max() - pd.Timedelta(days=5 * 365)]
        s = s[s.index.hour == 12]
        axs[1, j].scatter(s.shift(3).to_numpy(), s.to_numpy(), s=2, alpha=0.2)
        axs[1, j].set_xlabel("value 3 days earlier (12 UTC)"); axs[1, j].set_ylabel("value (12 UTC)")
    fig.suptitle("Weather (DWD Frankfurt 1420): recent data and the lagged regressor (all seasons mixed)")
    fig.tight_layout()
    P["pre_04_weather"] = fig
    return P


# --------------------------------------------------------------- Calibration --
def plots_calibration(B):
    P = {}
    hz = sorted(B.horizon.unique())

    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for j, tg in enumerate(TARGETS):
        d = B[B.target == tg]
        models = list(d.model.unique())
        for i, (c, nom) in enumerate([("in50", 0.5), ("in95", 0.95)]):
            ax = axs[i, j]
            cov = d.groupby(["model", "horizon"])[c].mean()
            for k, m in enumerate(models):
                ax.bar(np.arange(len(hz)) + (k - 0.5) * 0.4, cov[m].reindex(hz), width=0.4, color=COL[m], label=m)
            ax.axhline(nom, ls="--", c="k"); ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(hz)), hz); ax.set_title(f"{tg} - {int(nom * 100)}% interval", fontsize=9)
        axs[0, j].legend(fontsize=7)
    fig.suptitle("Backtest coverage of central intervals (dashed = nominal)")
    fig.tight_layout()
    P["cal_01_coverage"] = fig

    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, tg in zip(axs, TARGETS):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for m, g in B[B.target == tg].groupby("model"):
            ax.plot(QUANTILES, [np.mean(g.obs <= g[c]) for c in QCOLS], "o-", color=COL[m], label=m)
        ax.set_title(tg); ax.set_xlabel("nominal level"); ax.legend(fontsize=7)
    axs[0].set_ylabel("share of obs <= quantile")
    fig.suptitle("Reliability diagram (on the diagonal = calibrated)")
    fig.tight_layout()
    P["cal_02_reliability"] = fig

    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, tg in zip(axs, TARGETS):
        qs = B[B.target == tg].groupby(["model", "horizon"])["qs"].mean()
        for k, m in enumerate(qs.index.get_level_values(0).unique()):
            ax.bar(np.arange(len(hz)) + (k - 0.5) * 0.4, qs[m].reindex(hz), width=0.4, color=COL[m], label=m)
        ax.set_xticks(range(len(hz)), hz); ax.set_title(tg); ax.legend(fontsize=7)
    fig.suptitle("Mean quantile score by horizon (course metric, lower = better)")
    fig.tight_layout()
    P["cal_03_quantile_score"] = fig

    fig, axs = plt.subplots(4, 1, figsize=(12, 11))
    for ax, tg in zip(axs, TARGETS):
        d = B[(B.target == tg) & (B.model == "model") & (B.horizon == 36)].sort_values("origin")
        ax.vlines(d.origin, d["q0.025"], d["q0.975"], color="steelblue", alpha=0.4, lw=3)
        ax.vlines(d.origin, d["q0.25"], d["q0.75"], color="steelblue", lw=4)
        ax.scatter(d.origin, d.obs, s=10, c=np.where(d.in95, "k", "red"))
        ax.set_title(f"{LAB[tg]} - horizon 36 h")
    fig.suptitle("Backtest intervals vs. observations (red = outside the 95% interval)")
    fig.tight_layout()
    P["cal_04_backtest_36h"] = fig
    return P


# ---------------------------------------------------------- Current forecast --
def plot_current_forecast(FC, fd):
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for ax, tg in zip(axs.flat, TARGETS):
        d = FC[FC.target == tg]
        for k, (m, g) in enumerate(d.groupby("model")):
            x = np.arange(len(g)) + (k - 0.5) * 0.3
            ax.vlines(x, g["q0.025"], g["q0.975"], color=COL[m], lw=1)
            ax.vlines(x, g["q0.25"], g["q0.75"], color=COL[m], lw=5, label=m)
            ax.scatter(x, g["q0.5"], color="k", s=8, zorder=3)
        ax.set_xticks(range(6), [f"{h}h" for h in sorted(d.horizon.unique())])
        ax.set_title(LAB[tg]); ax.legend(fontsize=7)
    fig.suptitle(f"Forecast for {fd:%Y-%m-%d} (purple 'model' = submitted)")
    fig.tight_layout()
    return fig


def save_plots(P, plot_dir, pdf_path):
    os.makedirs(plot_dir, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for name, fig in P.items():
            fig.savefig(os.path.join(plot_dir, f"{name}.png"), bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
