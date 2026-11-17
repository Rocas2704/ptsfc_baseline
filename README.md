# PTSFC forecasting project

| Folder | Stage | Contents |
|---|---|---|
| [week1](week1/README.md) | Baseline | November 4 forecasts, backtests, submission, plots and diagnostics |
| [week2](week2/README.md) | Fourier terms and enhanced energy | November 11 comparisons, weather experiments, audits and validation |
| [week3](week3/README.md) | ECMWF | Forecast data, calibration, submissions, diagnostics, research reports and Overleaf exports |

Run commands from the repository root using the project Python environment. Shared code stays in `ptsfc/`, entry-point scripts and `config.py` stay at the root, and shared SMARD/DWD observation caches stay in `data/`. Checks and tests remain shared. Each week's README identifies its model functions, artifacts and commands.

The current pipeline includes enhanced energy comparisons and writes to `week2/output`. The ECMWF workflow writes to `week3/output/improved/<date>`.

[week_manifest.json](week_manifest.json) records the original path, new path and SHA-256 of all 271 moved files. Historical reports, ZIPs and run metadata retain their original contents and generation paths; consult the manifest for their current locations. The previous workspace README is preserved in [week3/docs/WORKFLOWS_LEGACY.md](week3/docs/WORKFLOWS_LEGACY.md) as historical documentation.
