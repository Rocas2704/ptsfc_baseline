"""
Port a Python del check_submission.R del curso (misma lógica y mensajes).
Uso:  python check_submission.py output/20261104_MiEquipo.csv
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

EXPECTED_COLS = ["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
TARGETS = ["demand", "res_demand", "temperature", "wind"]
HORIZON_VALS = [f"{h} hour" for h in (36, 40, 44, 60, 64, 68)]
TARGET_PLAUS = {"demand": (0, 200), "res_demand": (-200, 200), "temperature": (-50, 50), "wind": (0, 70)}
COLS_QUANTILES = EXPECTED_COLS[3:]


def check_df(path):
    """Devuelve True si el archivo pasa todos los checks (los WARNING no hacen fallar)."""
    stop = lambda m: (print(m), print("Stopping early..."), False)[2]  # noqa: E731
    try:
        df = pd.read_csv(path, sep=",", dtype=str, keep_default_na=False)
    except FileNotFoundError:
        return stop("File not found.")
    except Exception:  # noqa: BLE001
        return stop("Parse error or no data in given file.")
    if len(df) == 0:
        return stop("No data in given file.")

    print("File loaded! Start Checking...\n---------------------------")
    print("Checking the Columns...")
    cols = list(df.columns)
    if len(cols) != len(EXPECTED_COLS):
        return stop(f"Dataset contains {len(cols)} columns. Required are {len(EXPECTED_COLS)}")
    if set(cols) != set(EXPECTED_COLS):
        return stop(f"Dataset does not contain the required columns (or more).\n"
                    f"The missing columns are: {sorted(set(EXPECTED_COLS) - set(cols))}")
    if cols != EXPECTED_COLS:
        return stop(f"Columns not in correct order. Order should be: {EXPECTED_COLS}\nYour order is: {cols}")

    print("Checking type of columns...")
    fdate = pd.to_datetime(df["forecast_date"], format="%Y-%m-%d", errors="coerce")
    if fdate.isna().any():
        return stop("Could not parse Date in format YYYY-MM-DD")
    q = df[COLS_QUANTILES].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    if q.isna().any().any():
        i = int(np.where(q.isna().any(axis=1))[0][0])
        qq = COLS_QUANTILES[int(np.where(q.iloc[i].isna())[0][0])]
        return stop(f"Some quantile values are missing or not numeric! First occurrence: "
                    f"target '{df.target[i]}', horizon '{df.horizon[i]}', quantile '{qq}'.")
    if np.isinf(q.to_numpy()).any():
        return stop("Quantile values cannot be infinite.")

    print("Checking if the Dates make sense...")
    if fdate.nunique() > 1:
        return stop("forecast_date needs to be the same in all rows.")
    if fdate.iloc[0].date() < dt.date.today():
        print("----WARNING: Forecast date should not be in the past.\n")
    if fdate.iloc[0].isoweekday() != 3:
        print("----WARNING: Forecast date should be a Wednesday.\n")

    print("Checking targets...")
    if not df["target"].isin(TARGETS).all():
        return stop(f"Target column can only contain {', '.join(TARGETS)}. Check spelling.")
    for tg in TARGETS:
        sub = df[df.target == tg]
        if len(sub) != len(HORIZON_VALS):
            return stop(f"Exactly {len(HORIZON_VALS)} rows need to have target = {tg}")
        if list(sub.horizon) != HORIZON_VALS:
            return stop(f"Target {tg} horizons need to be (in this order): {', '.join(HORIZON_VALS)}")
        lo, hi = TARGET_PLAUS[tg]
        vals = q.loc[sub.index].to_numpy()
        if (vals < lo).any() or (vals > hi).any():
            print(f"----WARNING: Implausible values for {tg} detected. You may want to re-check.\n")

    print("Checking quantiles...")
    for i, row in enumerate(q.to_numpy()):
        if (np.diff(row) < 0).any():
            return stop(f"Predictive quantiles in row {i + 1} are not ordered correctly (need to be non-decreasing)")

    print("---------------------------\nLooks good!")
    return True


if __name__ == "__main__":
    sys.exit(0 if check_df(sys.argv[1] if len(sys.argv) > 1 else "submission_template.csv") else 1)
