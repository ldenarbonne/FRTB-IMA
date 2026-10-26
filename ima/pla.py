"""
P&L attribution test (MAR32.38–32.47) and desk-level backtesting (MAR32.18–32.20).

PLA: over the last 250 days, the Spearman rank correlation between HPL and RTPL and the Kolmogorov-Smirnov
distance between their empirical distributions place the desk in the green, amber or red zone.
Backtesting: VaR at 99 % and 97.5 % of the risk model (historical simulation on the preceding 250 days of
risk-theoretical P&L) against actual and hypothetical P&L; more than 12 exceptions at 99 % or 30 at 97.5 % over
250 days take the desk out of the internal model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import params as P


def spearman(h, r) -> float:
    return float(stats.spearmanr(h, r).statistic)


def ks(h, r) -> float:
    return float(stats.ks_2samp(h, r).statistic)


def zone(rho: float, d: float) -> str:
    t = P.PLA
    if rho < t["spearman_red"] or d > t["ks_red"]:
        return "red"
    if rho >= t["spearman_green"] and d <= t["ks_green"]:
        return "green"
    return "amber"


def pla_test(pnl: pd.DataFrame, n: int = 250) -> dict:
    w = pnl.iloc[-n:]
    rho, d = spearman(w["HPL"], w["RTPL"]), ks(w["HPL"], w["RTPL"])
    return {"Spearman": rho, "KS": d, "zone": zone(rho, d),
            "unexplained P&L / HPL vol": float((w["HPL"] - w["RTPL"]).std() / w["HPL"].std())}


def rolling_pla(pnl: pd.DataFrame, n: int = 250, step: int = 20) -> pd.DataFrame:
    rows = []
    for end in range(n, len(pnl) + 1, step):
        w = pnl.iloc[end - n:end]
        rows.append({"date": pnl.index[end - 1], "Spearman": spearman(w["HPL"], w["RTPL"]), "KS": ks(w["HPL"], w["RTPL"])})
    return pd.DataFrame(rows).set_index("date")


def var_series(rtpl: pd.Series, alpha: float, window: int = 250) -> pd.Series:
    """One-day VaR from historical simulation on the preceding `window` days of risk-model P&L (positive number)."""
    x = rtpl.to_numpy()
    v = np.full(len(x), np.nan)
    for t in range(window, len(x)):
        v[t] = -np.quantile(x[t - window:t], 1 - alpha)
    return pd.Series(v, index=rtpl.index)


def backtest(pnl: pd.DataFrame, n: int = 250) -> dict:
    out = {}
    for a in (0.99, 0.975):
        v = var_series(pnl["RTPL"], a).iloc[-n:]
        for k in ("APL", "HPL"):
            out[f"exceptions {a:.1%} {k}"] = int((pnl[k].iloc[-n:] < -v).sum())
        out[f"VaR {a:.1%}"] = v
    e99 = max(out["exceptions 99.0% APL"], out["exceptions 99.0% HPL"])
    e975 = max(out["exceptions 97.5% APL"], out["exceptions 97.5% HPL"])
    out["passes"] = e99 <= P.DESK_BT["exc_99_max"] and e975 <= P.DESK_BT["exc_975_max"]
    out["exc99"], out["exc975"] = e99, e975
    return out


def sweep(book, desk: str, cfg: dict, param: str, values) -> pd.DataFrame:
    """PLA metrics when one risk-model shortcoming varies, the others fixed."""
    from .data import pnl_series
    rows = []
    for v in values:
        r = pla_test(pnl_series(book, desk, {**cfg, param: v}))
        rows.append({param: v, **{k: r[k] for k in ("Spearman", "KS", "zone")}})
    return pd.DataFrame(rows)
