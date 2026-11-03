"""
PTSFC 2026/27 - BASELINE PIPELINE
Usage (from this directory):   python run_pipeline.py

Hallo
"""
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from check_submission import check_df
from config import CFG, QCOLS, TARGETS
from ptsfc.backtest import run_backtest, summarise
from ptsfc.data_energy import load_energy_data
from ptsfc.data_weather import load_weather_data
from ptsfc.models import forecast_all
from ptsfc.utils import msg, resolve_forecast_date

pd.set_option("display.width", 140)


def build_submission(FC, fd):
    sub = FC[FC.model == "model"].copy()
    sub[QCOLS] = np.sort(sub[QCOLS].to_numpy(float), axis=1).round(3)
    sub["forecast_date"] = f"{fd:%Y-%m-%d}"
    sub["target"] = pd.Categorical(sub["target"], categories=TARGETS, ordered=True)
    sub = sub.sort_values(["target", "horizon"])
    sub["target"] = sub["target"].astype(str)
    sub["horizon"] = sub["horizon"].astype(str) + " hour"
    return sub[["forecast_date", "target", "horizon"] + QCOLS].reset_index(drop=True)


def run_r_check(cfg, path):
    rscript = shutil.which("Rscript")
    if rscript and os.path.exists(cfg["r_check_script"]):
        out = subprocess.run([rscript, cfg["r_check_script"], path], capture_output=True, text=True)
        print(out.stdout.strip())
        if "Looks good!" not in out.stdout:
            raise SystemExit("The file did NOT pass the R check.")


def main(cfg=CFG):
    for d in (cfg["dir_data"], cfg["dir_output"], cfg["dir_archive"]):
        os.makedirs(d, exist_ok=True)
    fd = resolve_forecast_date(cfg)
    msg("==== PTSFC baseline | forecast_date = %s | team = %s ====", fd.date(), cfg["team_name"])

    # 1) Data
    ED = load_energy_data(cfg)
    W = load_weather_data(cfg)

    # 2) Forecast + submission file + official check
    FC = forecast_all(ED, W, cfg, fd)
    FC.to_csv(os.path.join(cfg["dir_output"], f"all_models_{fd:%Y%m%d}.csv"), index=False)
    sub = build_submission(FC, fd)
    out_csv = os.path.join(cfg["dir_output"], f"{fd:%Y%m%d}_{cfg['team_name']}.csv")
    sub.to_csv(out_csv, index=False)
    shutil.copy(out_csv, os.path.join(cfg["dir_archive"], os.path.basename(out_csv)))
    print(sub.to_string())
    if not check_df(out_csv):
        raise SystemExit("The file did NOT pass the check.")
    run_r_check(cfg, out_csv)
    msg("Submission written and checked: %s", out_csv)

    # 3) Backtest + plots
    P = {}
    if cfg["run_backtest"]:
        msg("Backtest over the last %d Wednesdays...", cfg["backtest_weeks"])
        B = run_backtest(ED, W, cfg, fd)
        B.to_csv(os.path.join(cfg["dir_output"], f"backtest_{fd:%Y%m%d}.csv"), index=False)
        print("\n==== Backtest summary ====")
        print(summarise(B).to_string())
        summarise(B, ("target", "horizon", "model")).to_csv(
            os.path.join(cfg["dir_output"], f"backtest_scores_{fd:%Y%m%d}.csv"), index=False)
    if cfg["make_plots"]:
        from ptsfc import plots
        P.update(plots.plots_preprocessing(ED, W))
        P["fc_01_current_forecast"] = plots.plot_current_forecast(FC, fd)
        if cfg["run_backtest"]:
            P.update(plots.plots_calibration(B))
        plots.save_plots(P, os.path.join(cfg["dir_output"], f"plots_{fd:%Y%m%d}"),
                         os.path.join(cfg["dir_output"], f"plots_{fd:%Y%m%d}.pdf"))
        msg("Plots saved in %s", cfg["dir_output"])
    msg("Done. File to submit: %s", out_csv)


if __name__ == "__main__":
    main()
