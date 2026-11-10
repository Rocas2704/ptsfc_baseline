"""Generate paired weather experiments from cached observations (offline by default)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG, HORIZONS, QCOLS
from ptsfc.backtest import summarise
from ptsfc.data_weather import load_weather_data
from ptsfc.models import forecast_weather, forecast_weather_enhanced
from ptsfc.utils import mean_qs, resolve_forecast_date

ROOT = Path(__file__).resolve().parent
VARIANTS = {'W1': 'lag2', 'W2a': 'full_lag2', 'W2b': 'fourier', 'W2c': 'fourier_lag1'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='Forecast Wednesday ending the backtest, e.g. 2026-11-04')
    parser.add_argument('--weeks', type=int, default=CFG['backtest_weeks'])
    parser.add_argument('--online', action='store_true', help='Refresh DWD data instead of using cache only')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/weather_experiment')
    args = parser.parse_args(argv)
    if args.weeks < 1:
        parser.error('--weeks must be positive')
    cfg = dict(CFG, forecast_date=args.date or CFG['forecast_date'], offline=not args.online,
               dir_data=str(ROOT / CFG['dir_data']))
    fd = resolve_forecast_date(cfg)
    if fd.day_of_week != 2:
        parser.error('--date must be a Wednesday')
    weather = load_weather_data(cfg)
    rows = []
    for k in range(1, args.weeks + 1):
        origin = fd - pd.Timedelta(weeks=k)
        for target in ('temperature', 'wind'):
            for horizon in HORIZONS:
                forecasts = [dict(forecast_weather(weather, target, origin, horizon, cfg)[0], model='W0')]
                for model, variant in VARIANTS.items():
                    row = forecast_weather_enhanced(weather, target, origin, horizon, cfg, variant)
                    forecasts.append(dict(row, model=model))
                # Exclude missing realizations for all specifications together.
                if all(np.isfinite(row['obs']) for row in forecasts):
                    rows.extend(dict(row, origin=origin) for row in forecasts)
    if not rows:
        raise ValueError('No realized forecast cases in the selected period')
    predictions = pd.DataFrame(rows)
    predictions['qs'] = mean_qs(predictions[QCOLS].to_numpy(float), predictions['obs'].to_numpy(float))
    predictions['error'] = predictions['obs'] - predictions['q0.5']
    predictions['in50'] = predictions['obs'].between(predictions['q0.25'], predictions['q0.75'])
    predictions['in95'] = predictions['obs'].between(predictions['q0.025'], predictions['q0.975'])
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    summarise(predictions).to_csv(args.output / 'summary.csv', index=False)
    summarise(predictions, ('target', 'horizon', 'model')).to_csv(args.output / 'by_horizon.csv', index=False)
    (args.output / 'run.json').write_text(json.dumps({
        'forecast_date': str(fd.date()), 'weeks': args.weeks, 'configuration': cfg,
        'variants': VARIANTS, 'created_utc': pd.Timestamp.now(tz='UTC').isoformat(),
    }, indent=2), encoding='utf-8')
    print(summarise(predictions).to_string(index=False))
    print(f'Saved experiment to {args.output}')


if __name__ == '__main__':
    main()
