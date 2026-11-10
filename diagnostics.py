
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import QUANTILES, QCOLS
from ptsfc.utils import mean_qs


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Saved-result diagnostics; run from repository root.')
    parser.add_argument('--input', default='output/backtest_20261104.csv')
    args = parser.parse_args(argv)
    # Load the existing 52-week backtest
    df = pd.read_csv(args.input)

    # Use the original OLS model only
    df = df[df["model"] == "model"].copy()
    df["origin"] = pd.to_datetime(df["origin"])
    df = df.sort_values("origin")

    df["error"] = df["obs"] - df["q0.5"]

    os.makedirs("output/diagnostics", exist_ok=True)

    # 1. Systematic forecast bias by target and horizon
    bias = df.groupby(["target", "horizon"])["error"].mean().unstack()
    bias.plot(kind="bar", figsize=(10, 5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Mean forecast error by target and horizon")
    plt.ylabel("Observation - predicted median")
    plt.tight_layout()
    plt.savefig("output/diagnostics/bias_by_horizon.png")
    plt.close()


    # 3. Rolling quantile score over time
    df["qs_recomputed"] = mean_qs(
        df[QCOLS].to_numpy(),
        df["obs"].to_numpy()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for ax, (target, group) in zip(axes.flat, df.groupby("target")):
        weekly = group.groupby("origin")["qs_recomputed"].mean()
        rolling = weekly.rolling(8, min_periods=4).mean()

        ax.plot(weekly.index, weekly, alpha=0.3, label="Weekly QS")
        ax.plot(rolling.index, rolling, linewidth=2, label="8-week rolling QS")
        ax.set_title(target)
        ax.set_ylabel("Mean quantile score")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/diagnostics/rolling_quantile_score.png")
    plt.close()

    print("Diagnostics completed.")
    print("Plots saved to output/diagnostics/")


if __name__ == '__main__':
    main()
