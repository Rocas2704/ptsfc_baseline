"""Compare saved probabilistic forecasts on matching cases and draw diagnostics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from config import QCOLS, QUANTILES, TARGETS
from ptsfc.utils import mean_qs

ROOT = Path(__file__).resolve().parent
KEYS = ['origin', 'target', 'horizon']


def prepare(frame):
    required = KEYS + ['model', 'obs'] + QCOLS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing forecast columns: {sorted(missing)}')
    df = frame.copy()
    df['origin'] = pd.to_datetime(df.origin)
    df = df.loc[df.obs.notna()].copy()
    if df[KEYS + ['model']].isna().any().any() or df.duplicated(KEYS + ['model']).any():
        raise ValueError('Missing identifiers or duplicate model/case rows')
    q = df[QCOLS].to_numpy(float)
    if not np.isfinite(q).all() or not np.isfinite(df.obs).all():
        raise ValueError('Nonfinite forecasts or observations')
    if (np.diff(q, axis=1) < -1e-9).any():
        raise ValueError('Crossing quantiles')
    if df.groupby(KEYS).obs.agg(lambda y: y.max() - y.min()).gt(1e-9).any():
        raise ValueError('Models do not share the same realized observations')
    df['qs'] = mean_qs(q, df.obs.to_numpy(float))
    df['error'] = df.obs - df['q0.5']
    df['in50'] = df.obs.between(df['q0.25'], df['q0.75'])
    df['in95'] = df.obs.between(df['q0.025'], df['q0.975'])
    return df


def summary(df, groups):
    return df.groupby(groups).agg(mean_QS=('qs', 'mean'), bias=('error', 'mean'),
                                  cov50=('in50', 'mean'), cov95=('in95', 'mean'),
                                  n=('qs', 'size')).reset_index()


def create_diagnostics(frame, output_dir, reference='model'):
    df = prepare(frame)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary(df, ['target', 'model']).to_csv(out / 'summary_all_available.csv', index=False)
    summary(df, ['target', 'horizon', 'model']).to_csv(out / 'summary_by_horizon.csv', index=False)
    matched_parts = []
    for _, group in df.groupby('target'):
        count = group.groupby(KEYS).model.nunique()
        common = count.loc[count.eq(group.model.nunique())].reset_index()[KEYS]
        matched_parts.append(group.merge(common, on=KEYS, validate='many_to_one'))
    matched = pd.concat(matched_parts, ignore_index=True)
    matched_summary = summary(matched, ['target', 'model'])
    matched_summary.to_csv(out / 'summary_common_cases.csv', index=False)
    reliability = []
    for (target, model), group in df.groupby(['target', 'model']):
        for col, tau in zip(QCOLS, QUANTILES):
            reliability.append(dict(target=target, model=model, nominal=tau,
                                    observed=(group.obs <= group[col]).mean(), n=len(group)))
    reliability = pd.DataFrame(reliability)
    reliability.to_csv(out / 'quantile_reliability.csv', index=False)
    reference_rows = df.loc[df.model.eq(reference), KEYS + ['qs']].rename(columns={'qs': 'reference_qs'})
    pairs = df.merge(reference_rows, on=KEYS, validate='many_to_one')
    pairs['gain'] = pairs.reference_qs - pairs.qs
    pairs.to_csv(out / 'paired_scores.csv', index=False)

    with PdfPages(out / 'backtest_diagnostics.pdf') as pdf:
        def save(fig, name):
            fig.tight_layout()
            fig.savefig(out / f'{name}.png', dpi=150)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, target in zip(axes.flat, TARGETS):
            s = matched_summary.loc[matched_summary.target.eq(target)]
            ax.bar(s.model, s.mean_QS)
            ax.tick_params(axis='x', rotation=50, labelsize=8)
            n = int(s.n.min()) if len(s) else 0
            ax.set(title=f'{target}: common cases (N={n}/model)', ylabel='Mean quantile score (lower is better)')
            ax.grid(axis='y', alpha=.2)
        save(fig, 'model_scores')

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, target in zip(axes.flat, TARGETS):
            for model, group in pairs.loc[pairs.target.eq(target)].groupby('model'):
                if model == reference:
                    continue
                weekly = group.groupby('origin').gain.mean().sort_index()
                ax.plot(weekly.index, weekly.cumsum(), label=model)
            ax.axhline(0, color='black', linewidth=.8)
            ax.set(title=target, ylabel=f'Cumulative weekly QS gain vs {reference}')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.autofmt_xdate()
        save(fig, 'cumulative_paired_gain')

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, target in zip(axes.flat, TARGETS):
            for model, group in df.loc[df.target.eq(target)].groupby('model'):
                weekly = group.groupby('origin').qs.mean().sort_index()
                ax.plot(weekly.index, weekly.rolling(8, min_periods=4).mean(), label=model)
            ax.set(title=target, ylabel='8-week rolling mean QS')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.autofmt_xdate()
        save(fig, 'rolling_quantile_score')

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, target in zip(axes.flat, TARGETS):
            for model, group in reliability.loc[reliability.target.eq(target)].groupby('model'):
                ax.plot(group.nominal, group.observed, marker='.', label=model)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set(title=target, xlabel='Nominal quantile probability', ylabel='Observed fraction below quantile')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        save(fig, 'quantile_reliability')

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, target in zip(axes.flat, TARGETS):
            for model, group in df.loc[df.target.eq(target)].groupby('model'):
                bias = group.groupby('horizon').error.mean()
                ax.plot(bias.index, bias, marker='.', label=model)
            ax.axhline(0, color='black', linewidth=.8)
            ax.set(title=target, xlabel='Course horizon', ylabel='Mean observation − median')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        save(fig, 'bias_by_horizon')

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, target in zip(axes.flat, TARGETS):
            for model, group in df.loc[df.target.eq(target)].groupby('model'):
                # Normalized week coverage is shown; no independence assumption for horizons.
                weekly = group.groupby('origin').in95.mean().sort_index()
                ax.plot(weekly.index, weekly.rolling(13, min_periods=8).mean(), label=model)
            ax.axhline(.95, color='black', linestyle='--', linewidth=.8)
            ax.set(title=target, ylabel='13-week rolling 95% coverage', ylim=(0, 1.05))
            ax.legend(fontsize=7)
        fig.autofmt_xdate()
        save(fig, 'coverage_over_time')
    print(f'Diagnostics saved: {out}', flush=True)
    return matched_summary


def plot_current(current, selected, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, target in zip(axes.flat, TARGETS):
        for model, group in current.loc[current.target.eq(target)].groupby('model'):
            group = group.sort_values('horizon')
            ax.plot(group.horizon, group['q0.5'], alpha=.45, label=model)
        group = selected.loc[selected.target.eq(target)].sort_values('horizon')
        ax.fill_between(group.horizon, group['q0.025'], group['q0.975'], color='navy', alpha=.10, label='Selected 95%')
        ax.fill_between(group.horizon, group['q0.25'], group['q0.75'], color='navy', alpha=.20, label='Selected 50%')
        ax.plot(group.horizon, group['q0.5'], 'o-', color='navy', linewidth=2, label='Selected median')
        ax.set(title=target, xlabel='Course horizon', ylabel='°C' if target == 'temperature' else 'm/s' if target == 'wind' else 'GWh')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'current_forecast.png', dpi=160)
    fig.savefig(out / 'current_forecast.pdf')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'week2/output/backtest_20261111.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'week2/output/diagnostics')
    parser.add_argument('--reference', default='model')
    parser.add_argument('--comparison-run', type=Path, help='Generate original-style baseline/selected plots from an improved run directory')
    args = parser.parse_args(argv)
    if args.comparison_run:
        create_comparison_plots(args.comparison_run)
    else:
        create_diagnostics(pd.read_csv(args.input), args.output, args.reference)


def create_comparison_plots(run_dir):
    """Reuse ptsfc.plots layouts with baseline and final selected model choices."""
    import json
    from ptsfc import plots
    from ptsfc.data_energy import load_energy_data
    from ptsfc.data_weather import load_weather_data
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    fd = pd.Timestamp(metadata['forecast_date'])
    backtest = prepare(pd.read_csv(run_dir / 'backtest_predictions.csv'))
    current = pd.read_csv(run_dir / 'current_predictions.csv')
    selected = pd.read_csv(run_dir / 'selected_predictions.csv')
    chosen, parts, live = {}, [], []
    for target in TARGETS:
        source = selected.loc[selected.target.eq(target), 'source_model'].unique()
        if len(source) != 1:
            raise ValueError(f'Expected one selected model for {target}')
        chosen[target] = source[0]
        candidate = backtest.loc[backtest.target.eq(target) & backtest.model.eq(source[0])].copy()
        baseline = backtest.loc[backtest.target.eq(target) & backtest.model.eq('baseline')].copy()
        common = candidate[KEYS].merge(baseline[KEYS], on=KEYS, validate='one_to_one')
        if common.empty:
            raise ValueError(f'No paired historical cases for {target}')
        candidate = candidate.merge(common, on=KEYS, validate='one_to_one')
        baseline = baseline.merge(common, on=KEYS, validate='one_to_one')
        candidate['source_model'], baseline['source_model'] = source[0], 'baseline'
        candidate['model'], baseline['model'] = 'selected', 'baseline'
        parts.extend([baseline, candidate])
        original = current.loc[current.target.eq(target) & current.model.eq('baseline')].copy()
        original['model'] = 'baseline'
        final = selected.loc[selected.target.eq(target)].copy()
        final['model'] = 'selected'
        live.extend([original, final])
    comparison = prepare(pd.concat(parts, ignore_index=True))
    live = pd.concat(live, ignore_index=True)
    directory = run_dir / f'plots_{fd:%Y%m%d}'
    directory.mkdir(exist_ok=True)
    comparison.to_csv(directory / 'comparison_backtest.csv', index=False)
    live.to_csv(directory / 'comparison_current.csv', index=False)
    summary(comparison, ['target', 'model']).to_csv(directory / 'comparison_scores.csv', index=False)
    cfg = dict(metadata['configuration'], offline=True)
    energy, weather = load_energy_data(cfg), load_weather_data(cfg)
    figures = plots.plots_preprocessing(energy, weather)
    figures.update(plots.plots_calibration(comparison, interval_model='selected'))
    # Separate baseline panel uses identical dates, observations and y scales.
    baseline_fig = plots.plots_calibration(comparison, interval_model='baseline')
    for name, figure in baseline_fig.items():
        if name == 'cal_04_backtest_36h':
            figures['cal_04_backtest_36h_baseline'] = figure
        else:
            plt.close(figure)
    selected_fig = figures['cal_04_backtest_36h']
    selected_fig.suptitle('Selected forecast: intervals vs. observations (red = outside 95%)')
    figures['cal_04_backtest_36h_baseline'].suptitle('Baseline: intervals vs. observations (red = outside 95%)')
    for left, right in zip(selected_fig.axes, figures['cal_04_backtest_36h_baseline'].axes):
        low, high = min(left.get_ylim()[0], right.get_ylim()[0]), max(left.get_ylim()[1], right.get_ylim()[1])
        left.set_ylim(low, high)
        right.set_ylim(low, high)
    figures['fc_01_current_forecast'] = plots.plot_current_forecast(live, fd, submitted_model='selected')
    # One additional page makes interval differences visible without switching pages.
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True, sharey='row')
    for row, target in enumerate(TARGETS):
        for col, model in enumerate(['baseline', 'selected']):
            ax = axes[row, col]
            group = comparison.loc[comparison.target.eq(target) & comparison.model.eq(model)
                                   & comparison.horizon.eq(36)].sort_values('origin')
            ax.vlines(group.origin, group['q0.025'], group['q0.975'], color=plots.COL[model], alpha=.35, lw=3)
            ax.vlines(group.origin, group['q0.25'], group['q0.75'], color=plots.COL[model], lw=4)
            ax.scatter(group.origin, group.obs, s=10, c=np.where(group.in95, 'black', 'red'), zorder=3)
            ax.set_title(f'{target}: {model} ({chosen[target] if model == "selected" else "original"})', fontsize=9)
            if col == 0:
                ax.set_ylabel(plots.LAB[target])
    fig.suptitle('Horizon 36h: baseline vs. selected on identical cases (shared scales)')
    fig.autofmt_xdate()
    fig.tight_layout()
    figures['cal_05_backtest_36h_comparison'] = fig
    pdf = run_dir / f'plots_{fd:%Y%m%d}.pdf'
    plots.save_plots(figures, str(directory), str(pdf))
    (directory / 'README.md').write_text(
        '# Original-style forecast comparisons\n\n'
        'Blue = original baseline; purple = selected submission model. Energy selections remain baseline.\n\n'
        'Calibration and scores use identical cases within each target; missing NWP cases are excluded from both models. '
        'Counts and plotted values are in comparison_backtest.csv and comparison_scores.csv.\n\n'
        'Historical selected rows use the final model choices retrospectively. This is a model comparison, '
        'not a backtest of a model-selection rule known before the evaluation year. '
        'The figures cover the full available evaluation sample; 45.2%/31.0% figures refer only to the final 13 weeks.\n\n'
        'Preprocessing plots use the current local observation caches. Comparison forecasts and outcomes come from '
        'the saved improved run, so the baseline can differ slightly from older saved plots after data revisions.\n',
        encoding='utf-8')
    print(f'Original-style comparison plots: {directory}\nCombined PDF: {pdf}', flush=True)
    return pdf


if __name__ == '__main__':
    main()
