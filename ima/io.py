"""CSV import of desk P&L series (PLA and backtesting) and of real-price observation logs (RFET)."""
from __future__ import annotations

import pandas as pd


def read_pnl(df: pd.DataFrame) -> dict:
    """Columns: date, desk, HPL, RTPL and optionally APL. Returns desk -> DataFrame(HPL, RTPL, APL)."""
    need = {"date", "desk", "HPL", "RTPL"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {', '.join(sorted(miss))}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "APL" not in df:
        df["APL"] = df["HPL"]
    out = {}
    for d, g in df.groupby("desk"):
        g = g.sort_values("date").set_index("date")[["HPL", "RTPL", "APL"]].astype(float)
        if len(g) < 250:
            raise ValueError(f"desk {d}: {len(g)} days, at least 250 are needed (500 for the backtesting VaR)")
        out[str(d)] = g
    return out


def read_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Columns: factor, date (one row per real-price observation)."""
    miss = {"factor", "date"} - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {', '.join(sorted(miss))}")
    out = df[["factor", "date"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.dropna()


def pnl_template(pnls: dict) -> pd.DataFrame:
    return pd.concat([p.assign(desk=d).rename_axis("date").reset_index() for d, p in pnls.items()])[
        ["date", "desk", "HPL", "RTPL", "APL"]].round({"HPL": 2, "RTPL": 2, "APL": 2})
