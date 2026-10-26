import numpy as np
import pandas as pd
import pytest

from ima import capital as C
from ima import data as D
from ima import io as IO
from ima import params as P
from ima import pla as PL
from ima import rfet as RF


@pytest.fixture(scope="module")
def book():
    return D.generate(7)


def test_zones():
    assert PL.zone(0.85, 0.05) == "green" and PL.zone(0.75, 0.05) == "amber" and PL.zone(0.85, 0.10) == "amber"
    assert PL.zone(0.65, 0.05) == "red" and PL.zone(0.90, 0.13) == "red"


def test_perfect_model_is_green(book):
    cfg = {"proxy_share": 0.0, "proxy_corr": 1.0, "pricer": "full", "missing": 0.0, "timing": 0.0}
    for d in book.desks:
        r = PL.pla_test(D.pnl_series(book, d, cfg))
        assert r["Spearman"] > 0.999 and r["KS"] <= 0.008 and r["zone"] == "green"


def test_shortcomings_degrade_pla(book):
    d = "Structured products"
    base = {"proxy_share": 0.0, "proxy_corr": 1.0, "pricer": "full", "missing": 0.0, "timing": 0.0}
    sw = PL.sweep(book, d, base, "missing", [0.0, 0.5, 1.0, 1.5])
    assert sw["Spearman"].is_monotonic_decreasing
    assert PL.pla_test(D.pnl_series(book, d, {**base, "pricer": "delta"}))["Spearman"] < 0.999


def test_default_story(book):
    z = {d: PL.pla_test(D.pnl_series(book, d, c))["zone"] for d, c in D.DEFAULT_CONFIG.items()}
    assert z["Rates G10"] == "green" and z["Structured products"] == "red" and "amber" in z.values()


def test_backtest_counts(book):
    p = D.pnl_series(book, "Rates G10", D.DEFAULT_CONFIG["Rates G10"])
    bt = PL.backtest(p)
    v = PL.var_series(p["RTPL"], 0.99).iloc[-250:]
    assert bt["exceptions 99.0% APL"] == int((p["APL"].iloc[-250:] < -v).sum())
    assert bt["exc975"] >= bt["exc99"]


def test_rfet_rules():
    asof = pd.Timestamp("2026-09-30")
    daily = pd.bdate_range(end=asof, periods=260)
    assert RF.test_factor(daily, asof)["modellable"]
    monthly = pd.date_range(end=asof, periods=24, freq="15D")            # 24 obs, regular → passes the 90-day rule
    r = RF.test_factor(monthly, asof)
    assert r["observations (12 m)"] == 24 and r["min obs in 90 days"] >= 4 and r["modellable"]
    gap = monthly[(monthly < asof - pd.Timedelta(days=200)) | (monthly > asof - pd.Timedelta(days=100))]
    gap = gap.append(pd.date_range(end=asof, periods=10, freq="D"))
    r = RF.test_factor(gap, asof)
    assert r["min obs in 90 days"] < 4 and not r["24 obs + 4 per 90 days"]
    burst = pd.bdate_range(end=asof, periods=100)                        # 100 obs in 5 months → passes on count alone
    assert RF.test_factor(burst, asof)["100 obs"] and RF.test_factor(burst, asof)["modellable"]


def test_es_and_lh_cascade(book):
    x = np.arange(1000) - 500.0
    assert np.isclose(C.es(x), -np.sort(x)[:25].mean())
    mod = pd.Series(True, index=book.factors.index)
    M = C.ten_day(book).iloc[-250:]
    tot, parts = C.es_lh(book, ["Rates G10"], D.DEFAULT_CONFIG, M, mod)
    H = P.LIQUIDITY_HORIZONS
    manual = np.sqrt(parts[0] ** 2 + sum((parts[j] * np.sqrt((H[j] - H[j - 1]) / 10)) ** 2 for j in range(1, 5)))
    assert np.isclose(tot, manual) and tot >= parts[0]


def test_imcc_and_ses(book):
    rf = RF.run(book.obs, book.factors.index, book.dates[-1])
    mod = rf["modellable"]
    desks = ["Rates G10", "Credit IG"]
    im = C.imcc(book, desks, D.DEFAULT_CONFIG, mod)
    assert np.isclose(im["IMCC"], 0.5 * im["unconstrained"]["ES"] + 0.5 * im["constrained sum"])
    assert im["unconstrained"]["ES"] >= im["unconstrained"]["ES_R,S"] - 1e-9
    ses = C.ses(book, desks, D.DEFAULT_CONFIG, list(rf.index[~mod]), im["stress start"])
    agg = C.aggregate_ses(ses)
    assert agg["SES"] <= agg["sum of stand-alone"] + 1e-9 and agg["idio credit"] > 0


def test_aggregation_formula():
    t = pd.DataFrame({"status": ["green", "amber", "SA"], "SA": [100.0, 100.0, 50.0], "DRC": [0.0, 10.0, 5.0]},
                     index=["a", "b", "c"])
    g = C.aggregate(t, imcc_val=40.0, ses_val=10.0, multiplier=1.5)
    assert np.isclose(g["C_A"], 70.0) and np.isclose(g["IMA_G,A"], 80.0) and np.isclose(g["k"], 0.25)
    assert np.isclose(g["PLA surcharge"], 0.25 * 120) and np.isclose(g["ACR"], min(80 + 30 + 50, 250))
    g2 = C.aggregate(t.assign(status="SA"), 40.0, 10.0, 1.5)
    assert np.isclose(g2["ACR"], 250.0)


def test_csv_import(book):
    pn = {d: D.pnl_series(book, d, c) for d, c in list(D.DEFAULT_CONFIG.items())[:2]}
    back = IO.read_pnl(IO.pnl_template(pn))
    assert set(back) == set(pn) and np.allclose(back["Rates G10"]["HPL"], pn["Rates G10"]["HPL"], atol=0.01)
    with pytest.raises(ValueError):
        IO.read_pnl(pd.DataFrame({"date": [], "desk": []}))


def test_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=900).run()
    assert not at.exception
