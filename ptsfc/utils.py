"""Utilities: dates, holidays, OLS and the quantile score."""
import datetime as dt

import numpy as np
import pandas as pd

from config import QUANTILES


def msg(text, *args):
    print(f"[{dt.datetime.now():%H:%M:%S}] " + (text % args if args else text), flush=True)


def wday_u(d):
    """Monday = 1 ... Sunday = 7."""
    if isinstance(d, (pd.Series, pd.DatetimeIndex)):
        return pd.DatetimeIndex(d).dayofweek + 1
    return pd.Timestamp(d).dayofweek + 1


def resolve_forecast_date(cfg):
    if cfg["forecast_date"] is not None:
        fd = pd.Timestamp(cfg["forecast_date"]).normalize()
    else:
        today = pd.Timestamp(dt.date.today())
        fd = today + pd.Timedelta(days=(3 - wday_u(today)) % 7)
    if wday_u(fd) != 3:
        msg("WARNING: forecast_date is not a Wednesday: %s", fd.date())
    return fd


# ---------------------------------------------------------------- Holidays --
def easter_date(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, month, day)


def special_days(years):
    """National holidays, large regional holidays and the Christmas/New Year period (24 Dec - 6 Jan)."""
    out = set()
    for y in years:
        e = easter_date(y)
        out |= {e + pd.Timedelta(days=k) for k in (-2, 1, 39, 50, 60)}   # Good Fri, Easter Mon, Ascension, Whit Mon, Corpus Christi
        out |= {pd.Timestamp(y, m, d) for m, d in [(5, 1), (8, 15), (10, 3), (10, 31), (11, 1)]}
        out |= {pd.Timestamp(y, 12, d) for d in range(24, 32)}
        out |= {pd.Timestamp(y, 1, d) for d in range(1, 7)}
    return out


def is_special_day(dates):
    dates = pd.DatetimeIndex(dates).normalize()
    days = special_days(range(dates.year.min() - 1, dates.year.max() + 2))
    return dates.isin(list(days))


# -------------------------------------------------------------- Regression --
def ols_quantiles(X, y, x_new, spread=1.0, taus=QUANTILES):
    """
    Multiple linear regression (OLS with intercept) + empirical residual quantiles:
        q_tau = x_new' beta  +  spread * quantile_tau(residuals)
    """
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    mu = float(np.concatenate([[1.0], np.atleast_1d(x_new)]) @ beta)
    return np.sort(mu + spread * np.quantile(resid, taus)), beta


# ------------------------------------------------------- Quantile score --
def quantile_score(q, y, tau):
    """Quantile score as in the course slides (factor 2)."""
    return np.where(q > y, 2 * (1 - tau) * (q - y), 2 * tau * (y - q))


def mean_qs(Q, y, taus=QUANTILES):
    Q, y = np.asarray(Q, float), np.asarray(y, float)
    return np.mean([quantile_score(Q[:, j], y, t) for j, t in enumerate(taus)], axis=0)
