"""
Synthetic trading book: risk factors, four years of daily factor moves (with a stress year), real-price
observation logs, six trading desks with non-linear positions, and the three P&L series of the FRTB tests:

* HPL – hypothetical P&L from the front-office pricers (full revaluation, every factor, true market moves);
* RTPL – risk-theoretical P&L from the risk model, whose shortcomings are set per desk:
  proxied factors, a simplified pricer (delta-gamma or delta only), factors missing from the model and a timing
  mismatch of the market data snapshot;
* APL – actual P&L = HPL + intraday trading, new deals and fees (used for backtesting).

Units: factor moves are in "units" with a daily volatility around 1 (outside the stress year); P&L in EUR thousands.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import params as P

# desk name -> list of (factor name, risk class, LH category, liquidity profile, in reduced set)
_DESKS = {
    "Rates G10": [
        ("EUR swap 2Y", "GIRR", "GIRR – specified currency", "liquid", True),
        ("EUR swap 10Y", "GIRR", "GIRR – specified currency", "liquid", True),
        ("EUR swap 30Y", "GIRR", "GIRR – specified currency", "liquid", True),
        ("USD SOFR 5Y", "GIRR", "GIRR – specified currency", "liquid", True),
        ("USD SOFR 30Y", "GIRR", "GIRR – specified currency", "liquid", True),
        ("GBP SONIA 10Y", "GIRR", "GIRR – specified currency", "liquid", False),
        ("EUR inflation 10Y", "GIRR", "GIRR – other", "moderate", False),
        ("EUR swaption vol 10Yx10Y", "GIRR", "GIRR – volatility", "moderate", True),
        ("EUR swaption vol 30Yx30Y", "GIRR", "GIRR – volatility", "sparse", False),
        ("EUR basis 3M/6M 40Y", "GIRR", "GIRR – other", "gap", False),
    ],
    "FX options": [
        ("EURUSD", "FX", "FX – specified pair", "liquid", True),
        ("USDJPY", "FX", "FX – specified pair", "liquid", True),
        ("EURGBP", "FX", "FX – specified pair", "liquid", True),
        ("USDTRY", "FX", "FX – other pair", "moderate", True),
        ("USDZAR", "FX", "FX – other pair", "moderate", False),
        ("EURUSD vol 1Y", "FX", "FX – volatility", "liquid", True),
        ("USDJPY vol 1Y", "FX", "FX – volatility", "moderate", True),
        ("USDTRY vol 1Y", "FX", "FX – volatility", "sparse", False),
        ("EURUSD 25D risk reversal 2Y", "FX", "FX – volatility", "borderline", False),
    ],
    "Credit IG": [
        ("iTraxx Main 5Y", "CSR", "CSR – corporate IG", "liquid", True),
        ("CDX IG 5Y", "CSR", "CSR – corporate IG", "liquid", True),
        ("OAT spread 10Y", "CSR", "CSR – sovereign IG", "liquid", True),
        ("BTP spread 10Y", "CSR", "CSR – sovereign IG", "liquid", True),
        ("Utility A 7Y", "CSR", "CSR – corporate IG", "moderate", True),
        ("Telecom BBB 10Y", "CSR", "CSR – corporate IG", "borderline", False),
        ("Real estate BBB- 5Y", "CSR", "CSR – corporate HY", "sparse", False),
        ("Private placement A 12Y", "CSR", "CSR – corporate IG", "gap", False),
        ("iTraxx Main vol", "CSR", "CSR – volatility", "sparse", False),
    ],
    "Equity derivatives": [
        ("EURO STOXX 50", "EQ", "EQ – large cap", "liquid", True),
        ("S&P 500", "EQ", "EQ – large cap", "liquid", True),
        ("CAC 40", "EQ", "EQ – large cap", "liquid", False),
        ("Small cap basket", "EQ", "EQ – small cap", "moderate", True),
        ("SX5E vol 1Y ATM", "EQ", "EQ – large cap volatility", "liquid", True),
        ("SX5E vol 5Y 80 %", "EQ", "EQ – large cap volatility", "borderline", False),
        ("SX5E dividend 2030", "EQ", "EQ – other", "gap", False),
        ("SX5E repo 3Y", "EQ", "EQ – other", "sparse", False),
        ("Small cap vol 1Y", "EQ", "EQ – small cap volatility", "sparse", False),
    ],
    "Commodities": [
        ("Brent 1M", "COM", "COM – energy & carbon", "liquid", True),
        ("TTF gas 1M", "COM", "COM – energy & carbon", "liquid", True),
        ("EUA carbon Dec", "COM", "COM – energy & carbon", "liquid", False),
        ("Gold", "COM", "COM – precious metals", "liquid", True),
        ("Copper 3M", "COM", "COM – other", "moderate", True),
        ("Brent vol 1Y", "COM", "COM – energy volatility", "moderate", True),
        ("TTF gas Cal-28", "COM", "COM – energy & carbon", "borderline", False),
        ("Cocoa 6M", "COM", "COM – other", "sparse", False),
    ],
    "Structured products": [
        ("EURO STOXX 50", "EQ", "EQ – large cap", "liquid", True),
        ("Nikkei 225", "EQ", "EQ – large cap", "liquid", False),
        ("SX5E vol 1Y ATM", "EQ", "EQ – large cap volatility", "liquid", True),
        ("SX5E vol 5Y 80 %", "EQ", "EQ – large cap volatility", "borderline", False),
        ("SX5E/NKY correlation", "EQ", "EQ – other", "sparse", False),
        ("SX5E dividend 2030", "EQ", "EQ – other", "gap", False),
        ("EUR swap 10Y", "GIRR", "GIRR – specified currency", "liquid", True),
        ("Long-dated SX5E vol 10Y", "EQ", "EQ – other", "sparse", False),
    ],
}

# default risk-model shortcomings per desk (the PLA "story")
DEFAULT_CONFIG = {
    "Rates G10": dict(proxy_share=0.0, proxy_corr=0.95, pricer="full", missing=0.05, timing=0.05),
    "FX options": dict(proxy_share=0.0, proxy_corr=0.95, pricer="full", missing=0.10, timing=0.30),
    "Credit IG": dict(proxy_share=0.45, proxy_corr=0.80, pricer="full", missing=0.55, timing=0.05),
    "Equity derivatives": dict(proxy_share=0.0, proxy_corr=0.90, pricer="delta-gamma", missing=0.50, timing=0.10),
    "Commodities": dict(proxy_share=0.15, proxy_corr=0.90, pricer="full", missing=0.10, timing=0.22),
    "Structured products": dict(proxy_share=0.50, proxy_corr=0.60, pricer="delta", missing=0.75, timing=0.10),
}

OBS_RATE = {"liquid": 230, "moderate": 45, "sparse": 14, "gap": 40, "borderline": 34}   # observations per year


@dataclass
class Book:
    dates: pd.DatetimeIndex
    factors: pd.DataFrame            # name, class, LH category, LH, profile, reduced
    moves: pd.DataFrame              # daily factor moves (dates × factors)
    stress_start: pd.Timestamp       # stress year used in the simulation (for reference)
    obs: pd.DataFrame                # real-price observations: factor, date
    desks: dict                      # desk -> dict(factors, delta, gamma, cubic, basis_beta)
    noise: dict = field(default_factory=dict)   # fixed random draws for the risk-model shortcomings


def generate(seed: int = 7, end: str = "2026-09-30", years: int = 4) -> Book:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=260 * years)
    T = len(dates)
    names, rows = [], []
    for desk, fl in _DESKS.items():
        for f, cls, cat, prof, red in fl:
            if f not in names:
                names.append(f)
                rows.append({"factor": f, "risk class": cls, "category": cat, "liquidity horizon": P.LH_CATEGORY[cat],
                             "profile": prof, "reduced set": red})
    fac = pd.DataFrame(rows).set_index("factor")
    F = len(fac)
    classes = sorted(fac["risk class"].unique())
    # factor model: global + class + idiosyncratic, Student t innovations, stress year with higher vol
    load_g = rng.uniform(0.2, 0.5, F)
    load_c = rng.uniform(0.3, 0.6, F)
    cls_idx = fac["risk class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    t = lambda size: rng.standard_t(5, size) / np.sqrt(5 / 3)
    g = t(T)
    c = t((T, len(classes)))
    e = t((T, F))
    idio = np.sqrt(np.clip(1 - load_g ** 2 - load_c ** 2, 0.05, None))
    X = load_g * g[:, None] + load_c * c[:, cls_idx] + idio * e
    s0 = 260 // 2                                   # stress year: second half of year 1 to first half of year 2
    vol = np.ones(T)
    vol[s0:s0 + 260] = np.concatenate([np.linspace(1.4, 2.6, 60), np.full(140, 2.6), np.linspace(2.6, 1.3, 60)])
    sig = rng.uniform(0.7, 1.3, F)
    X = X * vol[:, None] * sig
    moves = pd.DataFrame(X, index=dates, columns=fac.index)

    # observation logs (real prices) over the whole history
    obs = []
    for f, r in fac.iterrows():
        lam = OBS_RATE[r["profile"]] / 260
        hit = rng.random(T) < lam
        if r["profile"] == "gap":           # long blackout in the last year (illiquid summer, data vendor change)
            b = T - rng.integers(120, 200)
            hit[b:b + 110] = False
        if r["profile"] == "borderline":    # enough quotes overall, but a thin quarter in the last year
            b = T - rng.integers(90, 240)
            hit[b:b + 60] &= rng.random(60) < 0.6
        obs += [(f, d) for d in dates[hit]]
    obs = pd.DataFrame(obs, columns=["factor", "date"])

    desks = {}
    for k, (desk, fl) in enumerate(_DESKS.items()):
        fs = [x[0] for x in fl]
        n = len(fs)
        scale = {"Structured products": 1.3, "Equity derivatives": 1.1}.get(desk, 1.0)
        delta = rng.normal(0, 60, n) * scale
        gamma = rng.normal(0, 6, n) * scale
        if desk in ("Equity derivatives", "Structured products", "FX options"):
            gamma = -np.abs(gamma) * 1.6           # short optionality
        illiquid = np.array([book_prof != "liquid" for book_prof in [x[3] for x in fl]])
        small = np.where(illiquid, 0.35, 1.0)          # smaller positions in illiquid factors
        delta, gamma = delta * small, gamma * small
        cubic = np.zeros(n)
        if desk in ("Structured products", "Equity derivatives", "Credit IG"):
            cubic = rng.normal(0, 0.9, n) * scale * small
        nb = 3
        desks[desk] = {"factors": fs, "delta": delta, "gamma": gamma, "cubic": cubic,
                       "basis": rng.standard_t(4, (T, nb)) * vol[:, None] / np.sqrt(2), "basis_beta": rng.normal(0, 1, nb)}
    noise = {"proxy": rng.standard_normal((T, F)), "timing": rng.standard_normal((T, F)),
             "apl": rng.standard_t(4, (T, len(desks))), "proxy_rank": rng.random(F)}
    return Book(dates, fac, moves, dates[s0], obs, desks, noise)


# -------------------------------------------------------------------------------------------- P&L functions
def desk_pnl(book: Book, desk: str, X: np.ndarray, pricer: str = "full", cols=None) -> np.ndarray:
    """P&L of a desk for factor moves X (n × desk factors). `cols`: boolean mask of the factors shocked."""
    d = book.desks[desk]
    if cols is not None:
        X = np.where(cols, X, 0.0)
    out = X @ d["delta"]
    if pricer in ("full", "delta-gamma"):
        out = out + 0.5 * (X ** 2) @ d["gamma"]
    if pricer == "full":
        out = out + (X ** 3) @ d["cubic"] / 6
    return out


def _desk_moves(book: Book, desk: str) -> np.ndarray:
    return book.moves[book.desks[desk]["factors"]].to_numpy()


def hpl(book: Book, desk: str) -> pd.Series:
    """Front-office hypothetical P&L: full revaluation on all factors + basis factors absent from the risk model."""
    X = _desk_moves(book, desk)
    main = desk_pnl(book, desk, X, "full")
    d = book.desks[desk]
    basis = d["basis"] @ d["basis_beta"]
    return pd.Series(main, index=book.dates), pd.Series(basis / basis.std() * main.std(), index=book.dates)


def model_moves(book: Book, desk: str, cfg: dict) -> np.ndarray:
    """Factor moves as seen by the risk model: proxies and timing mismatch."""
    fs = book.desks[desk]["factors"]
    idx = [book.factors.index.get_loc(f) for f in fs]
    X = _desk_moves(book, desk).copy()
    sd = X.std(axis=0)
    # proxied factors: the least liquid ones first (rank by profile, then fixed random order)
    order = {"gap": 0, "sparse": 1, "borderline": 2, "moderate": 3, "liquid": 4}
    rank = sorted(range(len(fs)), key=lambda i: (order[book.factors.loc[fs[i], "profile"]], book.noise["proxy_rank"][idx[i]]))
    k = int(round(cfg["proxy_share"] * len(fs)))
    rho = cfg["proxy_corr"]
    for i in rank[:k]:
        X[:, i] = rho * X[:, i] + np.sqrt(1 - rho ** 2) * sd[i] * book.noise["proxy"][:, idx[i]]
    if cfg["timing"] > 0:
        X = X + cfg["timing"] * sd * book.noise["timing"][:, idx]
    return X


def pnl_series(book: Book, desk: str, cfg: dict) -> pd.DataFrame:
    main, basis = hpl(book, desk)
    H = main + cfg["missing"] * basis
    R = desk_pnl(book, desk, model_moves(book, desk, cfg), cfg["pricer"])
    k = list(book.desks).index(desk)
    A = H + 0.12 * main.std() * book.noise["apl"][:, k] + 0.02 * main.std()      # intraday, new trades, fees
    return pd.DataFrame({"HPL": H, "RTPL": R, "APL": A}, index=book.dates)
