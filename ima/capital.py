"""
Capital under the internal models approach (MAR33):

* expected shortfall at 97.5 % on 10-day moves with the liquidity-horizon cascade (MAR33.4)
      ES = sqrt( ES_T(P)² + Σ_{j≥2} ( ES_T(P, j) · sqrt((LH_j − LH_{j−1}) / T) )² ),  T = 10 days,
  where ES_T(P, j) shocks only the modellable factors whose liquidity horizon is at least LH_j;
* stress calibration with a reduced set of factors (MAR33.5): ES = ES_R,S × max(ES_F,C / ES_R,C, 1), the stress
  period being the 250-day window that maximises the reduced-set ES;
* IMCC = ρ · ES(all classes) + (1 − ρ) · Σ ES(class), ρ = 0.5 (MAR33.15–33.16);
* stressed expected shortfall of non-modellable factors (MAR33.17):
      SES = sqrt(Σ ISES²_idio credit) + sqrt(Σ ISES²_idio equity) + sqrt((ρ Σ SES_k)² + (1 − ρ²) Σ SES_k²), ρ = 0.6;
* aggregation with the standardised approach, the PLA surcharge and the backtesting multiplier (MAR33.41–33.45):
      C_A = max(IMCC + SES, m_c · IMCC_avg + SES_avg);  IMA_G,A = C_A + DRC_G,A;
      surcharge = k · max(0, SA_G,A − IMA_G,A),  k = 0.5 · Σ_{amber} SA_i / Σ_{green+amber} SA_i;
      ACR = min(IMA_G,A + surcharge + C_U, SA_all) + max(0, IMA_G,A − SA_G,A).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P
from .data import Book, desk_pnl

IDIO_CREDIT = {"Utility A 7Y", "Telecom BBB 10Y", "Real estate BBB- 5Y", "Private placement A 12Y"}
IDIO_EQUITY: set = set()


def ten_day(book: Book) -> pd.DataFrame:
    return book.moves.rolling(P.BASE_HORIZON).sum().dropna()


def es(x: np.ndarray, alpha: float = P.ES_ALPHA) -> float:
    """Expected shortfall of a P&L vector (positive = loss)."""
    loss = -np.sort(np.asarray(x))
    k = max(int(np.ceil(round(len(loss) * (1 - alpha), 9))), 1)
    return float(np.sort(loss)[-k:].mean())


def portfolio_pnl(book: Book, desks: list, cfgs: dict, M: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    out = np.zeros(len(M))
    for d in desks:
        fs = book.desks[d]["factors"]
        out += desk_pnl(book, d, M[fs].to_numpy(), cfgs[d]["pricer"], cols=mask[fs].to_numpy())
    return out


def es_lh(book, desks, cfgs, M, base_mask: pd.Series) -> tuple[float, list]:
    """Liquidity-adjusted ES for the factors in `base_mask`; returns the total and the ES per horizon bucket."""
    lh = book.factors["liquidity horizon"]
    H = P.LIQUIDITY_HORIZONS
    parts = []
    for j, h in enumerate(H):
        e = es(portfolio_pnl(book, desks, cfgs, M, base_mask & (lh >= h)))
        parts.append(e)
    tot = parts[0] ** 2 + sum((parts[j] * np.sqrt((H[j] - H[j - 1]) / P.BASE_HORIZON)) ** 2 for j in range(1, len(H)))
    return float(np.sqrt(tot)), parts


def stress_window(book, desks, cfgs, modellable: pd.Series, M=None, window: int = 250, step: int = 10):
    M = ten_day(book) if M is None else M
    red = modellable & book.factors["reduced set"]
    best, rows = None, []
    for s in range(0, len(M) - window + 1, step):
        v, _ = es_lh(book, desks, cfgs, M.iloc[s:s + window], red)
        rows.append({"window end": M.index[s + window - 1], "ES reduced set": v})
        if best is None or v > best[0]:
            best = (v, s)
    return best[1], pd.DataFrame(rows).set_index("window end")


def imcc(book, desks, cfgs, modellable: pd.Series, stress_start: int | None = None, window: int = 250) -> dict:
    M = ten_day(book)
    cur = M.iloc[-window:]
    if stress_start is None:
        stress_start, scan = stress_window(book, desks, cfgs, modellable, M, window)
    else:
        scan = None
    stress = M.iloc[stress_start:stress_start + window]
    red = modellable & book.factors["reduced set"]
    cls = book.factors["risk class"]

    def stressed(mask):
        fc, _ = es_lh(book, desks, cfgs, cur, mask)
        rc, _ = es_lh(book, desks, cfgs, cur, mask & red)
        rs, parts = es_lh(book, desks, cfgs, stress, mask & red)
        ratio = fc / rc if rc > 0 else 1.0
        return {"ES_F,C": fc, "ES_R,C": rc, "ES_R,S": rs, "ratio": ratio, "ES": rs * max(ratio, 1.0), "parts": parts}

    allc = stressed(modellable)
    by_cls = {c: stressed(modellable & (cls == c)) for c in sorted(cls.unique())
              if any((cls[book.desks[d]["factors"]] == c).any() for d in desks)}
    unconstrained = allc["ES"]
    constrained = sum(v["ES"] for v in by_cls.values())
    val = P.IMCC_RHO * unconstrained + (1 - P.IMCC_RHO) * constrained
    return {"IMCC": val, "unconstrained": allc, "by class": by_cls, "constrained sum": constrained,
            "stress window": (M.index[stress_start], M.index[min(stress_start + window, len(M)) - 1]),
            "stress start": stress_start, "scan": scan,
            "reduced-set coverage": allc["ES_R,C"] / allc["ES_F,C"] if allc["ES_F,C"] > 0 else np.nan}


def ses(book, desks, cfgs, nmrf: list, stress_start: int, window: int = 250) -> pd.DataFrame:
    """Stress scenario per NMRF: ES at 97.5 % of the P&L from that factor alone over the stress period, scaled to its
    liquidity horizon (at least 20 days)."""
    M = ten_day(book).iloc[stress_start:stress_start + window]
    rows = []
    for f in nmrf:
        holders = [d for d in desks if f in book.desks[d]["factors"]]
        if not holders:
            continue
        mask = pd.Series(False, index=book.factors.index)
        mask[f] = True
        lh = max(int(book.factors.loc[f, "liquidity horizon"]), 20)
        loss = es(portfolio_pnl(book, holders, cfgs, M, mask)) * np.sqrt(lh / P.BASE_HORIZON)
        kind = "idiosyncratic credit" if f in IDIO_CREDIT else ("idiosyncratic equity" if f in IDIO_EQUITY else "other")
        rows.append({"factor": f, "desks": ", ".join(holders), "liquidity horizon (≥ 20)": lh, "type": kind,
                     "stress loss": loss})
    return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame(
        columns=["desks", "liquidity horizon (≥ 20)", "type", "stress loss"])


def aggregate_ses(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"SES": 0.0, "idio credit": 0.0, "idio equity": 0.0, "other": 0.0, "sum of stand-alone": 0.0}
    s = t["stress loss"]
    ic = float(np.sqrt((s[t["type"] == "idiosyncratic credit"] ** 2).sum()))
    ie = float(np.sqrt((s[t["type"] == "idiosyncratic equity"] ** 2).sum()))
    o = s[t["type"] == "other"]
    oth = float(np.sqrt((P.SES_RHO * o.sum()) ** 2 + (1 - P.SES_RHO ** 2) * (o ** 2).sum()))
    return {"SES": ic + ie + oth, "idio credit": ic, "idio equity": ie, "other": oth, "sum of stand-alone": float(s.sum())}


def bank_backtest(pnls: dict, desks: list) -> dict:
    from .pla import var_series
    if not desks:
        return {"exceptions": 0, "plus factor": 0.0, "multiplier": P.MULTIPLIER_BASE}
    tot = sum(pnls[d] for d in desks)
    v = var_series(tot["RTPL"], 0.99).iloc[-250:]
    ea = int((tot["APL"].iloc[-250:] < -v).sum())
    eh = int((tot["HPL"].iloc[-250:] < -v).sum())
    e = max(ea, eh)
    return {"exceptions APL": ea, "exceptions HPL": eh, "exceptions": e, "plus factor": P.plus_factor(e),
            "multiplier": P.MULTIPLIER_BASE + P.plus_factor(e), "VaR": v, "pnl": tot.iloc[-250:]}


def aggregate(desk_tab: pd.DataFrame, imcc_val: float, ses_val: float, multiplier: float, avg_ratio: float = 1.0) -> dict:
    """desk_tab: index desk, columns 'status' (green / amber / SA), 'SA', 'DRC'."""
    ga = desk_tab["status"].isin(["green", "amber"])
    am = desk_tab["status"] == "amber"
    imcc_avg, ses_avg = imcc_val * avg_ratio, ses_val * avg_ratio
    CA = max(imcc_val + ses_val, multiplier * imcc_avg + ses_avg) if ga.any() else 0.0
    ima = CA + float(desk_tab.loc[ga, "DRC"].sum())
    sa_ga = float(desk_tab.loc[ga, "SA"].sum())
    cu = float(desk_tab.loc[~ga, "SA"].sum())
    sa_all = float(desk_tab["SA"].sum())
    k = 0.5 * float(desk_tab.loc[am, "SA"].sum()) / sa_ga if sa_ga > 0 else 0.0
    sur = k * max(0.0, sa_ga - ima)
    acr = min(ima + sur + cu, sa_all) + max(0.0, ima - sa_ga)
    return {"IMCC": imcc_val, "SES": ses_val, "multiplier": multiplier, "C_A": CA, "DRC (G, A)": float(desk_tab.loc[ga, "DRC"].sum()),
            "IMA_G,A": ima, "SA_G,A": sa_ga, "C_U": cu, "k": k, "PLA surcharge": sur, "SA all desks": sa_all,
            "ACR": acr, "binding": "SA (all desks)" if ima + sur + cu > sa_all else "IMA"}


def default_sa_drc(book, cfgs, modellable_all: pd.Series) -> pd.DataFrame:
    """Illustrative standardised and DRC charges per desk (to be replaced by the bank's SA engine output)."""
    M = ten_day(book)
    cur = M.iloc[-250:]
    mult = {"Rates G10": 1.6, "FX options": 1.5, "Credit IG": 2.0, "Equity derivatives": 1.8, "Commodities": 1.7,
            "Structured products": 2.4}
    drc = {"Credit IG": 0.30, "Equity derivatives": 0.08, "Structured products": 0.10}
    rows = []
    for d in book.desks:
        all_f = pd.Series(True, index=book.factors.index)
        e, _ = es_lh(book, [d], cfgs, cur, all_f)
        sa = 5.5 * e * mult[d]
        rows.append({"desk": d, "SA": round(sa, -1), "DRC": round(sa * drc.get(d, 0.0), -1)})
    return pd.DataFrame(rows).set_index("desk")
