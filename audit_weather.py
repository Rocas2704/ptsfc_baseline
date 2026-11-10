"""Audit W0, W1, W2a, W2b and W2c without modifying the pipeline.

Run from repository root after `python experiment_weather.py`:
    python audit_weather.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CFG, QCOLS
from ptsfc.data_weather import load_weather_data
from ptsfc.utils import mean_qs

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'output/weather_experiment/predictions.csv'
OUT = ROOT / 'output/weather_audit'
MODELS = ['W0', 'W1', 'W2a', 'W2b', 'W2c']
TARGETS = ['temperature', 'wind']
HORIZONS = [36, 40, 44, 60, 64, 68]
COMPARISONS = [('W0', m) for m in MODELS[1:]] + [('W1', 'W2a'), ('W2a', 'W2b'), ('W2c', 'W2b')]
BOOTSTRAP_REPS = 3000
BLOCK_WEEKS = 4


def paired_bootstrap(differences, rng):
    """Exploratory circular block-bootstrap CI of a mean weekly score gain."""
    n = len(differences)
    if n < 2:
        return (np.nan, np.nan)
    block = min(BLOCK_WEEKS, n)
    starts = rng.integers(0, n, size=(BOOTSTRAP_REPS, int(np.ceil(n / block))))
    indexes = (starts[:, :, None] + np.arange(block)) % n
    indexes = indexes.reshape(BOOTSTRAP_REPS, -1)[:, :n]
    estimates = differences[indexes].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975])


def main(input_path=INPUT, output_dir=OUT):
    INPUT, OUT = Path(input_path), Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(f'{INPUT} not found. First run: python experiment_weather.py')

    B = pd.read_csv(INPUT)
    B['origin'] = pd.to_datetime(B['origin'])
    keys = ['origin', 'target', 'horizon']
    if B.duplicated(keys + ['model']).any():
        raise ValueError('Duplicate forecasts for same origin, target, horizon and model.')
    if set(B['model']) != set(MODELS):
        raise ValueError(f'Expected models {MODELS}; found {sorted(B["model"].unique())}')
    if set(B['target']) != set(TARGETS):
        raise ValueError('Unexpected target set.')
    if set(B['horizon']) != set(HORIZONS):
        raise ValueError('Unexpected horizon set.')
    count = B.groupby(keys)['model'].nunique()
    if not count.eq(len(MODELS)).all():
        raise ValueError('Not every observed target/origin/horizon has all five models.')
    if not np.isfinite(B[QCOLS + ['obs']].to_numpy(float)).all():
        raise ValueError('Nonfinite quantiles or realized outcomes present.')
    if (np.diff(B[QCOLS].to_numpy(float), axis=1) < -1e-9).any():
        raise ValueError('Crossing quantiles detected.')
    if B.groupby(keys)['obs'].agg(lambda z: z.max() - z.min()).gt(1e-9).any():
        raise ValueError('Realized outcomes differ across models.')
    B['qs_audit'] = mean_qs(B[QCOLS].to_numpy(float), B['obs'].to_numpy(float))
    if 'qs' in B and not np.allclose(B['qs'].to_numpy(float), B['qs_audit'], atol=1e-8):
        raise ValueError('Stored QS does not agree with recomputed QS.')
    print(f'PASS: {len(B):,} scored forecasts, all models paired, correct quantiles and observations.')

    # Reconstruct the model's timestamp rules and valid training-row counts.
    cfg = dict(CFG, offline=True, dir_data=str(ROOT / CFG['dir_data']))
    W = load_weather_data(cfg)
    audit = []
    for (fd, target, horizon), _ in B.groupby(keys, sort=False):
        s = W[target]
        fd = pd.Timestamp(fd)
        cutoff = fd.tz_localize('UTC') - pd.Timedelta(hours=1)
        t0 = s.loc[:cutoff].last_valid_index()
        if t0 is None:
            raise ValueError(f'No prior weather data for {target} at {fd}')
        T = (fd + pd.Timedelta(days=1)).tz_localize('UTC') + pd.Timedelta(hours=int(horizon))
        k0 = int(np.ceil((T - t0) / pd.Timedelta(days=1)))
        lag1 = pd.Timedelta(days=k0)
        lag2 = pd.Timedelta(days=k0 + 1)
        n_days = int(365.25 * cfg['weather_train_years'])
        Td_full = T - pd.to_timedelta(np.arange(k0, n_days), unit='D')
        distance = np.abs(Td_full.day_of_year - T.day_of_year)
        Td_season = Td_full[np.minimum(distance, 365 - distance) <= cfg['weather_season_days']]

        for model in MODELS:
            Td = Td_season if model in ('W0', 'W1') else Td_full
            uses_lag2 = model in ('W1', 'W2a', 'W2b')
            y = s.reindex(Td).to_numpy(float)
            x1 = s.reindex(Td - lag1).to_numpy(float)
            valid = np.isfinite(y) & np.isfinite(x1)
            if uses_lag2:
                x2 = s.reindex(Td - lag2).to_numpy(float)
                valid &= np.isfinite(x2)
            last_predictor = T - lag1
            xnew_valid = np.isfinite(s.get(last_predictor, np.nan))
            if uses_lag2:
                xnew_valid &= np.isfinite(s.get(T - lag2, np.nan))
            fallback = model != 'W0' and (valid.sum() < 30 or not xnew_valid)
            audit.append({
                'origin': fd, 'target': target, 'horizon': horizon, 'model': model,
                'n_train': int(valid.sum()), 'fallback': bool(fallback),
                'train_y_after_cutoff': bool((Td[valid] > cutoff).any()),
                'train_x_after_cutoff': bool(((Td[valid] - lag1) > cutoff).any()),
                'forecast_x_after_cutoff': bool(last_predictor > cutoff),
                'train_start': Td[valid].min() if valid.any() else pd.NaT,
                'train_end': Td[valid].max() if valid.any() else pd.NaT,
            })

    A = pd.DataFrame(audit)
    A.to_csv(OUT / 'training_audit.csv', index=False)
    print('\n=== TRAINING AND LEAKAGE AUDIT ===')
    print(A.groupby(['target', 'model']).agg(
        min_n=('n_train', 'min'), median_n=('n_train', 'median'),
        max_n=('n_train', 'max'), fallback=('fallback', 'sum'),
        future_train_y=('train_y_after_cutoff', 'sum'),
        future_train_x=('train_x_after_cutoff', 'sum'),
        future_forecast_x=('forecast_x_after_cutoff', 'sum')
    ).to_string())
    leakage_columns = ['train_y_after_cutoff', 'train_x_after_cutoff', 'forecast_x_after_cutoff']
    if A[leakage_columns].any().any():
        raise ValueError('Timestamp leakage flagged; investigate before using results.')

    # Aggregate six horizon scores *per Wednesday* before comparing.
    wide = B.pivot(index=keys, columns='model', values='qs_audit').reset_index()
    out = []
    rng = np.random.default_rng(20261107)
    for target in TARGETS:
        weekly = wide.loc[wide['target'].eq(target)].groupby('origin')[MODELS].mean().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        for reference, candidate in COMPARISONS:
            gain = (weekly[reference] - weekly[candidate]).to_numpy()
            lo, hi = paired_bootstrap(gain, rng)
            mean_gain = gain.mean()
            out.append({
                'target': target, 'reference': reference, 'candidate': candidate,
                'n_weeks': len(gain), 'reference_QS': weekly[reference].mean(),
                'candidate_QS': weekly[candidate].mean(), 'mean_gain': mean_gain,
                'gain_percent': 100 * mean_gain / weekly[reference].mean(),
                'weeks_better': int((gain > 0).sum()),
                'CI_low': lo, 'CI_high': hi,
            })
            # One combined plot for main baseline comparisons; other contrasts are in CSV.
            if reference == 'W0':
                ax.plot(weekly.index, np.cumsum(gain), label=f'{candidate} vs W0')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set(title=f'{target}: cumulative QS advantage over W0',
               xlabel='Forecast Wednesday', ylabel='Cumulative QS gain (positive = better)')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT / f'{target}_cumulative_comparison.png', dpi=170)
        plt.close(fig)

    R = pd.DataFrame(out)
    R.to_csv(OUT / 'paired_comparison.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    main(args.input, args.output)
