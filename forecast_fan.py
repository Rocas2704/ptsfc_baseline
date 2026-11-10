
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import CFG
from ptsfc.data_energy import load_energy_data
from ptsfc.data_weather import load_weather_data
from ptsfc.models import ENERGY_SLOTS


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Saved-result diagnostics; run from repository root.')
    parser.add_argument('--date', default='2026-11-04')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args(argv)
    # --------------------------------------------------
    # SETTINGS
    # --------------------------------------------------
    FORECAST_DATE = args.date
    HISTORY_DAYS = 5
    TARGETS = ["demand", "res_demand", "temperature", "wind"]
    HORIZONS = [36, 40, 44, 60, 64, 68]

    fd = pd.Timestamp(FORECAST_DATE)
    DAY = pd.Timedelta(days=1)
    HOUR = pd.Timedelta(hours=1)

    os.makedirs("output/diagnostics", exist_ok=True)

    # Use the SAVED Week 1 forecasts, not today's model code.
    forecast_file = f"output/all_models_{fd:%Y%m%d}.csv"

    FC = pd.read_csv(forecast_file)
    FC = FC[FC["model"] == "model"].copy()

    # Downloaded/cached observations
    cfg = dict(CFG, offline=True)
    ED = load_energy_data(cfg)
    W = load_weather_data(cfg)


    def berlin_naive(timestamp):
        """Convert an aware timestamp to local Berlin plotting time."""
        return pd.Timestamp(timestamp).tz_convert(
            "Europe/Berlin"
        ).tz_localize(None)


    def prepare_target(target):
        """Historical information, target times and realized values."""

        f = FC[FC["target"] == target].copy()
        f = f.sort_values("horizon")

        if target in ("demand", "res_demand"):

            # Match the model's historical information cutoff:
            # latest complete day available before Wednesday.
            lcd = ED.last_complete_day(target, fd - DAY)

            hist = ED.E[["time_utc", "date", target]].dropna()
            hist = hist[hist["date"] <= lcd].copy()

            hist["time"] = (
                hist["time_utc"]
                .dt.tz_convert("Europe/Berlin")
                .dt.tz_localize(None)
            )

            cutoff = hist["time"].max()

            times = []
            observations = []

            for h in f["horizon"]:
                offset, hour_end = ENERGY_SLOTS[int(h)]
                target_day = fd + offset * DAY

                # SMARD hour_end=12 corresponds to the hour
                # beginning at 11:00 local time.
                local_time = (
                    target_day + pd.Timedelta(hours=hour_end - 1)
                )

                times.append(local_time)

                obs = ED.wide[target][hour_end].get(
                    target_day, np.nan
                )
                observations.append(obs)

        else:
            # Same historical cutoff logic as forecast_weather().
            s = W[target]

            t0 = s.loc[
                :fd.tz_localize("UTC") - HOUR
            ].last_valid_index()

            cutoff = berlin_naive(t0)

            hist = s.loc[:t0].dropna().reset_index()
            hist["time"] = (
                hist["time"]
                .dt.tz_convert("Europe/Berlin")
                .dt.tz_localize(None)
            )

            times = []
            observations = []

            for h in f["horizon"]:
                T = (
                    (fd + DAY).tz_localize("UTC")
                    + int(h) * HOUR
                )

                times.append(berlin_naive(T))
                observations.append(s.get(T, np.nan))

        hist = hist[
            (hist["time"] >= cutoff - HISTORY_DAYS * DAY)
            & (hist["time"] <= cutoff)
        ].sort_values("time")

        f["plot_time"] = times
        f["actual"] = observations

        return hist, f, cutoff


    # --------------------------------------------------
    # FAN CHARTS
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    summary = []

    for ax, target in zip(axes.flat, TARGETS):

        hist, f, cutoff = prepare_target(target)

        # Historical observations
        ax.plot(
            hist["time"],
            hist[target],
            color="black",
            linewidth=1.3,
            label="Available observations"
        )

        # Last available historical observation
        ax.axvline(
            cutoff,
            color="gray",
            linestyle="--",
            linewidth=1.3
        )

        # Six separate prediction distributions
        for _, row in f.iterrows():

            t = row["plot_time"]

            # 95% interval
            ax.vlines(
                t, row["q0.025"], row["q0.975"],
                color="#93c5fd",
                linewidth=8,
                alpha=0.55
            )

            # 50% interval
            ax.vlines(
                t, row["q0.25"], row["q0.75"],
                color="#2563eb",
                linewidth=8,
                alpha=0.55
            )

            # Forecast median
            ax.plot(
                t, row["q0.5"],
                marker="o",
                color="#1d4ed8",
                markersize=5,
                linestyle="None"
            )

            # Realized outcome, if available
            actual = row["actual"]

            if pd.notna(actual):

                inside95 = (
                    row["q0.025"] <= actual <= row["q0.975"]
                )

                ax.scatter(
                    t, actual,
                    color="black" if inside95 else "red",
                    marker="x",
                    s=75,
                    linewidths=2,
                    zorder=10
                )

                summary.append({
                    "target": target,
                    "horizon": row["horizon"],
                    "actual": actual,
                    "median": row["q0.5"],
                    "error": actual - row["q0.5"],
                    "inside50": (
                        row["q0.25"] <= actual <= row["q0.75"]
                    ),
                    "inside95": inside95
                })

        # Labels
        titles = {
            "demand": "Electricity demand",
            "res_demand": "Residual demand",
            "temperature": "2 m temperature",
            "wind": "10 m wind speed"
        }

        units = {
            "demand": "GWh",
            "res_demand": "GWh",
            "temperature": "°C",
            "wind": "m/s"
        }

        ax.set_title(titles[target])
        ax.set_ylabel(units[target])
        ax.set_xlabel("Date / time (Europe/Berlin)")
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=30)

    # Shared legend
    legend = [
        Line2D([0], [0], color="black",
               label="Historical observations"),
        Line2D([0], [0], color="gray", linestyle="--",
               label="Information cutoff"),
        Patch(facecolor="#93c5fd", alpha=0.55,
              label="95% prediction interval"),
        Patch(facecolor="#2563eb", alpha=0.55,
              label="50% prediction interval"),
        Line2D([0], [0], marker="o", color="#1d4ed8",
               linestyle="None", label="Forecast median"),
        Line2D([0], [0], marker="x", color="black",
               linestyle="None", label="Observed (within 95%)"),
        Line2D([0], [0], marker="x", color="red",
               linestyle="None", label="Observed (outside 95%)")
    ]

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(
        f"Week 1 — Probabilistic forecast evaluation ({fd:%Y-%m-%d})",
        fontsize=15
    )

    plt.tight_layout(rect=[0, 0.11, 1, 0.95])

    path = f"output/diagnostics/forecast_evaluation_{fd:%Y%m%d}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")

    # --------------------------------------------------
    # EVALUATION SUMMARY
    # --------------------------------------------------
    results = pd.DataFrame(summary)

    if not results.empty:
        print("\nForecast evaluation:")
        print(results.to_string(index=False))

        print("\nCoverage by target:")
        print(
            results.groupby("target")[["inside50", "inside95"]]
            .mean()
            .round(3)
        )

        results.to_csv(
            f"output/diagnostics/forecast_evaluation_{fd:%Y%m%d}.csv",
            index=False
        )
    else:
        print("No realized target observations available yet.")

    print(f"\nSaved: {path}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
