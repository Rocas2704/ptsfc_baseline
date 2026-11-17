"""
MULTI-MODEL ENSEMBLE for temperature and wind (v3) - EMOS on several weather centres.

Idea
----
Every weather centre (ECMWF, DWD, NOAA, Meteo-France, UK Met Office, Canada, Japan, ...) runs its
own model. Their errors are partly independent, so a weighted combination is better than any
single model, and the DISAGREEMENT between them says how uncertain the situation is.
We do not copy any model's output: we learn from the past forecast-vs-observation pairs
  (1) how much to trust each source             -> combination weights w_m
  (2) how to correct the common bias            -> a, b
  (3) how the uncertainty grows with the spread -> c, d
This is EMOS / non-homogeneous Gaussian regression (Gneiting et al. 2005, Mon. Wea. Rev.),
which was originally developed exactly for multi-centre ensembles.

Model (for one target time T, one variable, lead day k = 2 Fri / 3 Sat)
------------------------------------------------------------------------
    members  x_1..x_M = forecasts of the M sources for T, issued k days before
                        (+ the run one day older, k+1, if ens_lagged: "time-lagged ensemble")
    mean     mu    = a + b * sum_m w_m x_m (+ g * last observed same-hour value, optional)
             w = softmax(theta)  -> weights >= 0 that sum to 1 (stable with collinear sources),
                                    ridge penalty on theta shrinks towards equal weights
    spread   sigma^2 = exp(c) + exp(d) * S^2,   S^2 = variance of the members at T
    Y ~ N(mu, sigma^2)   (wind: on the square-root scale, quantiles squared back)

Parameters are estimated by minimising the mean CRPS (closed form for the normal, analytic
gradient -> fast enough for the daily backtest). The CRPS is the integral of the quantile score
over all levels, so this directly targets the competition metric.

Training rows: the same clock hour, last ens_window_days days + the same season (+/- ens_season_days)
in earlier years, observed before the submission time t0 (no look-ahead).
A source is used only if it is available for T and in >= ens_min_coverage of the training rows;
with fewer than ens_min_models sources the model returns None and models.py falls back.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

from config import QUANTILES
from .data_nwp import fetch_point
from .utils import msg

DAY, H = pd.Timedelta(days=1), pd.Timedelta(hours=1)
STATION_VARS = (("temperature", "temperature_2m"), ("wind", "wind_speed_10m"))
SQRT_PI = np.sqrt(np.pi)


# ===================================================================== DATA ==
def ens_max_lead(cfg):
    return cfg["nwp_max_lead_day"] + (1 if cfg["ens_lagged"] else 0)


def load_station_ensemble(cfg, end):
    """
    dict(temperature=DataFrame, wind=DataFrame); columns "<model>|d<k>" (k = 0..max lead).
    Sources that fail are skipped; raises if fewer than ens_min_models remain.
    """
    L = ens_max_lead(cfg)
    out = {n: [] for n, _ in STATION_VARS}
    for model in cfg["ens_models"]:
        for name, var in STATION_VARS:
            try:
                d = fetch_point(cfg, "station", cfg["station_lat"], cfg["station_lon"], var, end, model=model, max_lead=L)
                out[name].append(d.rename(columns=lambda c, m=model: f"{m}|d{c.split('_d')[-1]}"))
            except Exception as e:  # noqa: BLE001
                msg("WARNING: ensemble source %s (%s) skipped (%s)", model, var, e)
    E = {n: pd.concat(v, axis=1).sort_index() for n, v in out.items() if v}
    if not E:
        raise RuntimeError("no ensemble source available")

    # coverage report: share of hours with a lead-day-2 forecast since the start of the archive
    k = min(2, cfg["nwp_max_lead_day"])
    rep = {}
    for n, D in E.items():
        D = D[(D.index >= pd.Timestamp(cfg["nwp_start_date"], tz="UTC")) & (D.index <= pd.Timestamp.now(tz="UTC"))]
        for m in models_in(D):
            c = f"{m}|d{k}"
            if c in D:
                v = D[c]
                rep.setdefault(m, {})[n] = v.notna().mean()
                rep[m]["first"] = v.first_valid_index()
    R = pd.DataFrame(rep).T
    msg("Ensemble sources (coverage of d%d forecasts since %s):\n%s", k, cfg["nwp_start_date"], R.to_string())
    if min(len(models_in(D)) for D in E.values()) < cfg["ens_min_models"]:
        raise RuntimeError(f"fewer than {cfg['ens_min_models']} ensemble sources")
    return E


def models_in(D):
    return list(dict.fromkeys(c.split("|")[0] for c in D.columns))


def ens_members(E, var, k, times, cfg, live=False):
    """
    Member matrix (len(times) x M) for lead day k (+ k+1 if ens_lagged), with member names and source names.
    live=True: the main member (lead k) is filled with the latest run where the lead-k value is missing.
    """
    D, times = E[var], pd.DatetimeIndex(times)
    leads = [k, k + 1] if cfg["ens_lagged"] else [k]
    cols, names, srcs = [], [], []
    for m in models_in(D):
        for j, kk in enumerate(leads):
            c = f"{m}|d{kk}"
            if c not in D:
                continue
            v = D[c].reindex(times).to_numpy(float)
            if live and j == 0 and np.isnan(v).any() and f"{m}|d0" in D:
                v = np.where(np.isnan(v), D[f"{m}|d0"].reindex(times).to_numpy(float), v)
            cols.append(v)
            names.append(c)
            srcs.append(m)
    X = np.column_stack(cols) if cols else np.empty((len(times), 0))
    return X, names, np.array(srcs)


# ===================================================================== EMOS ==
def _softmax(t):
    e = np.exp(t - t.max())
    return e / e.sum()


def _unpack(p, M, equal, has_z):
    a, b, i = p[0], p[1], 2
    th = np.zeros(M) if equal else p[i:i + M]
    i += 0 if equal else M
    g = p[i] if has_z else 0.0
    i += int(has_z)
    return a, b, th, g, p[i], p[i + 1]


def _loss_grad(p, X, y, z, S2, sy, ridge, equal):
    n, M = X.shape
    has_z = z is not None
    a, b, th, g, c, d = _unpack(p, M, equal, has_z)
    w = _softmax(th)
    fbar = X @ w
    mu = a + b * fbar + (g * z if has_z else 0.0)
    ec, ed = np.exp(c), np.exp(d)
    sig = np.sqrt(ec + ed * S2)
    u = (y - mu) / sig
    Phi, phi = ndtr(u), np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    crps = sig * (u * (2 * Phi - 1) + 2 * phi - 1 / SQRT_PI)
    loss = crps.mean() / sy + ridge * np.sum(th ** 2) / M

    dmu = (1 - 2 * Phi) / (n * sy)                       # d loss / d mu_i
    dsig = (2 * phi - 1 / SQRT_PI) / (n * sy)            # d loss / d sigma_i
    grad = [dmu.sum(), dmu @ fbar]
    if not equal:
        grad += list(b * w * ((dmu[:, None] * (X - fbar[:, None])).sum(axis=0)) + 2 * ridge * th / M)
    if has_z:
        grad.append(dmu @ z)
    grad += [dsig @ (ec / (2 * sig)), dsig @ (ed * S2 / (2 * sig))]
    return loss, np.array(grad)


def emos_fit(X, y, ridge=0.05, equal=False, z=None):
    """
    Fit the EMOS model (see module docstring) by minimising the mean CRPS.
    X: n x M member forecasts, y: observations, z: optional extra regressor (persistence).
    Returns a dict with the parameters, the weights and the in-sample mu / sigma.
    """
    n, M = X.shape
    S2 = X.var(axis=1, ddof=1) if M > 1 else np.zeros(n)
    sy = max(np.std(y), 1e-6)

    # start values: OLS of y on the equal-weight mean (+ z), residual variance split between c and d
    Z = np.column_stack([np.ones(n), X.mean(axis=1)] + ([z] if z is not None else []))
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    v = max(np.var(y - Z @ beta), 1e-6)
    c0, d0 = np.log(0.5 * v), np.log(max(0.5 * v / max(S2.mean(), 1e-9), 1e-8))
    p0 = [beta[0], beta[1]] + ([] if equal else [0.0] * M) + ([beta[2]] if z is not None else []) + [c0, d0]
    bounds = [(None, None), (-1.0, 3.0)] + ([] if equal else [(-6.0, 6.0)] * M) \
        + ([(-2.0, 2.0)] if z is not None else []) + [(-20.0, 10.0), (-20.0, 10.0)]
    p0 = np.clip(p0, [lo if lo is not None else -np.inf for lo, _ in bounds],
                 [hi if hi is not None else np.inf for _, hi in bounds])

    res = minimize(_loss_grad, p0, args=(X, y, z, S2, sy, ridge, equal), jac=True,
                   method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500))
    a, b, th, g, c, d = _unpack(res.x, M, equal, z is not None)
    w = _softmax(th)
    mu = a + b * (X @ w) + (g * z if z is not None else 0.0)
    sig = np.sqrt(np.exp(c) + np.exp(d) * S2)
    return dict(a=a, b=b, w=w, g=g, c=c, d=d, mu=mu, sig=sig, S2=S2, n=n, converged=res.success,
                crps=float(res.fun - ridge * np.sum(th ** 2) / M) * sy)


def emos_predict(fit, x_new, z_new=None):
    x_new = np.asarray(x_new, float)
    S2 = x_new.var(ddof=1) if len(x_new) > 1 else 0.0
    mu = fit["a"] + fit["b"] * (x_new @ fit["w"]) + (fit["g"] * z_new if z_new is not None else 0.0)
    return float(mu), float(np.sqrt(np.exp(fit["c"]) + np.exp(fit["d"]) * S2)), float(np.sqrt(S2))


def emos_quantiles(fit, mu, sig, y, dist="normal", spread=1.0, taus=QUANTILES):
    if dist == "empirical":                              # standardised residuals of the training rows
        r = (y - fit["mu"]) / fit["sig"]
        return np.sort(mu + spread * sig * np.quantile(r, taus))
    return mu + spread * sig * ndtri(np.asarray(taus))


# ================================================================= FORECAST ==
def train_times(T, t0, cfg):
    """Past target times at the same clock hour, observed before t0: last N days + same season."""
    n = int((T - pd.Timestamp(cfg["nwp_start_date"], tz="UTC")) / DAY)
    Td = T - pd.to_timedelta(np.arange(1, max(n, 2)) * 24, unit="h")
    Td = Td[Td <= t0]
    dist = np.abs(Td.dayofyear - T.dayofyear)
    return Td[(Td > t0 - cfg["ens_window_days"] * DAY) | (np.minimum(dist, 365 - dist) <= cfg["ens_season_days"])]


def _drop_for_rows(X, oky, use, srcs, cfg):
    """
    Every member has >= ens_min_coverage, but if their gaps fall on different days the complete rows
    (all members present) can be too few. Greedy fix: while there are fewer than ens_min_rows complete
    rows, drop the member whose removal gains the most rows (keeping >= ens_min_models sources).
    """
    ok = np.isfinite(X)
    n_rows = lambda u: int((oky & ok[:, u].all(axis=1)).sum())   # noqa: E731
    use = use.copy()
    while n_rows(use) < cfg["ens_min_rows"]:
        best, best_n = None, -1
        for j in np.flatnonzero(use):
            u = use.copy()
            u[j] = False
            if len(set(srcs[u])) >= cfg["ens_min_models"] and n_rows(u) > best_n:
                best, best_n = j, n_rows(u)
        if best is None:
            break
        use[best] = False
    return use


def forecast_ensemble(W, E, var, fd, T, t0, cfg, live=False, coefs=None, horizon=None, equal=False):
    """Quantiles of the multi-model EMOS forecast for target time T, or None if it cannot be fitted."""
    if E is None or var not in E:
        return None
    k = int(min(max((T.tz_localize(None).normalize() - fd).days, 1), cfg["nwp_max_lead_day"]))
    sqrt = var == "wind" and cfg["wind_sqrt"]
    f = (lambda v: np.sqrt(np.clip(v, 0, None))) if sqrt else (lambda v: np.asarray(v, float))

    Td = train_times(T, t0, cfg)
    X, names, srcs = ens_members(E, var, k, Td, cfg)
    x_new = ens_members(E, var, k, [T], cfg, live=live)[0][0]
    y = f(W[var].reindex(Td).to_numpy(float))
    if X.shape[1] == 0:
        return None

    # member selection: available for T and well covered in the training rows
    oky = np.isfinite(y)
    cov = np.isfinite(X[oky]).mean(axis=0) if oky.any() else np.zeros(X.shape[1])
    use = np.isfinite(x_new) & (cov >= cfg["ens_min_coverage"])
    if len(set(srcs[use])) < cfg["ens_min_models"]:
        return None
    use = _drop_for_rows(X, oky, use, srcs, cfg)          # sources with different gaps -> keep enough complete rows
    rows = oky & np.isfinite(X[:, use]).all(axis=1)

    z = z_new = None
    if cfg["ens_obs_lag"]:                               # persistence: last observed same clock hour
        lag = int(np.ceil((T - t0) / H / 24)) * 24 * H
        z, z_new = f(W[var].reindex(Td - lag).to_numpy(float)), float(f(W[var].get(T - lag, np.nan)))
        rows &= np.isfinite(z)
        if not np.isfinite(z_new):
            z = z_new = None
    if rows.sum() < cfg["ens_min_rows"]:
        return None

    Xu = f(X[rows][:, use])
    fit = emos_fit(Xu, y[rows], cfg["ens_ridge"], equal, None if z is None else z[rows])
    mu, sig, spread_ens = emos_predict(fit, f(x_new[use]), z_new)
    q = emos_quantiles(fit, mu, sig, y[rows], cfg["ens_dist"], cfg["weather_spread"][var])
    q = np.clip(q, 0, None) ** 2 if sqrt else q

    if coefs is not None:
        base = dict(origin=fd, target=var, horizon=horizon, model="model", n=fit["n"], crps_train=fit["crps"])
        sd = Xu.std(axis=0, ddof=1)
        coefs.append(dict(base, term="intercept", beta=fit["a"]))
        coefs.append(dict(base, term="scale_b", beta=fit["b"]))
        coefs += [dict(base, term=f"w_{nm}", beta=wj, impact=fit["b"] * wj * sj)
                  for nm, wj, sj in zip(np.array(names)[use], fit["w"], sd)]
        if z is not None:
            coefs.append(dict(base, term="obs_lag", beta=fit["g"]))
        coefs.append(dict(base, term="sigma_const", beta=float(np.exp(fit["c"] / 2))))
        coefs.append(dict(base, term="sigma_per_spread", beta=float(np.exp(fit["d"] / 2))))
        coefs.append(dict(base, term="ens_sd_today", beta=spread_ens))
    return np.sort(q)