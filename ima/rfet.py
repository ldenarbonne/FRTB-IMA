"""
Risk factor eligibility test (MAR31.12–31.13): a risk factor is modellable if, over the previous 12 months, it
has at least 24 real-price observations with no 90-day period containing fewer than 4 of them, or at least 100
observations. Otherwise it is a non-modellable risk factor (NMRF), capitalised through a stress scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P


def test_factor(dates: pd.Series | pd.DatetimeIndex, asof: pd.Timestamp) -> dict:
    R = P.RFET
    start = asof - pd.Timedelta(days=R["lookback_days"])
    d = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates)))).normalize()
    d = d[(d > start) & (d <= asof)]
    n = len(d)
    # minimum number of observations in any 90-day window [s, s + 89 days] fully inside the 12 months
    days = pd.date_range(start + pd.Timedelta(days=1), asof - pd.Timedelta(days=R["window_days"] - 1), freq="D")
    if n:
        pos = np.searchsorted(d.values, days.values)
        pos_end = np.searchsorted(d.values, (days + pd.Timedelta(days=R["window_days"])).values)
        counts = pos_end - pos
        worst = int(counts.min())
        worst_start = days[int(counts.argmin())]
    else:
        worst, worst_start = 0, days[0]
    crit_a = n >= R["min_obs_with_gap_rule"] and worst >= R["min_obs_per_window"]
    crit_b = n >= R["min_obs_alone"]
    return {"observations (12 m)": n, "min obs in 90 days": worst, "weakest window start": worst_start,
            "24 obs + 4 per 90 days": crit_a, "100 obs": crit_b, "modellable": crit_a or crit_b}


def run(obs: pd.DataFrame, factors: pd.Index, asof: pd.Timestamp) -> pd.DataFrame:
    g = obs.groupby("factor")["date"]
    rows = []
    for f in factors:
        dts = g.get_group(f) if f in g.groups else pd.Series([], dtype="datetime64[ns]")
        rows.append({"factor": f, **test_factor(dts, asof)})
    return pd.DataFrame(rows).set_index("factor")


def monthly_counts(obs: pd.DataFrame, factor: str, asof: pd.Timestamp) -> pd.Series:
    d = pd.DatetimeIndex(obs.loc[obs["factor"] == factor, "date"])
    d = d[(d > asof - pd.Timedelta(days=365)) & (d <= asof)]
    return pd.Series(1, index=d).resample("ME").sum().reindex(
        pd.date_range(asof - pd.Timedelta(days=365), asof, freq="ME"), fill_value=0)
