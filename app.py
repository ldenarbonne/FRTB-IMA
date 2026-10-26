"""FRTB IMA Lab – P&L attribution test, risk factor eligibility, non-modellable risk factors and IMA capital."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ima import capital as C
from ima import data as D
from ima import io as IO
from ima import params as P
from ima import pla as PL
from ima import rfet as RF

st.set_page_config(page_title="FRTB IMA Lab", page_icon="📐", layout="wide")
ZC = {"green": "#3b8c5a", "amber": "#d99a2b", "red": "#b8433a", "SA": "#8a8f98"}
PRICERS = ["full", "delta-gamma", "delta"]


def h(df):
    return min(35 * (len(df) + 1) + 3, 900)


def text(df: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        f = fmt.get(c)
        if f is not None:
            out[c] = [("–" if (v is None or (isinstance(v, float) and np.isnan(v))) else f.format(v)) for v in df[c]]
    return out


def chip(z):
    return {"green": "🟢 green", "amber": "🟠 amber", "red": "🔴 red", "SA": "⚪ SA"}.get(z, z)


@st.cache_data
def book(seed):
    return D.generate(seed)


@st.cache_data
def pnl(seed, desk, cfg_items):
    return D.pnl_series(book(seed), desk, dict(cfg_items))


@st.cache_data(show_spinner="Computing expected shortfall with liquidity horizons and the stress period…")
def run_imcc(seed, desks, cfg_key, mod_key):
    b = book(seed)
    cfgs = {d: dict(c) for d, c in cfg_key}
    mod = pd.Series(dict(mod_key))
    return C.imcc(b, list(desks), cfgs, mod)


@st.cache_data
def run_ses(seed, desks, cfg_key, nmrf, start):
    b = book(seed)
    cfgs = {d: dict(c) for d, c in cfg_key}
    return C.ses(b, list(desks), cfgs, list(nmrf), start)


@st.cache_data
def default_sa(seed, cfg_key):
    b = book(seed)
    return C.default_sa_drc(b, {d: dict(c) for d, c in cfg_key}, None)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    seed = st.number_input("Seed of the synthetic bank", 0, 9999, 7)
    b = book(seed)
    asof = b.dates[-1]
    st.caption(f"{len(b.desks)} desks, {len(b.factors)} risk factors, {len(b.dates)} business days to "
               f"{asof:%d %b %Y}; real-price observations: {len(b.obs):,}.")
    st.header("Capital settings")
    avg_ratio = st.slider("60-day average IMCC and SES / latest value", 0.7, 1.3, 1.0, 0.05,
                          help="C_A = max(IMCC + SES at t−1, m_c × 60-day average IMCC + 60-day average SES). The averages "
                               "are proxied by the latest value times this ratio.")
    extra_plus = st.slider("Qualitative add-on to the multiplier", 0.0, 0.5, 0.0, 0.05)

if "cfg" not in st.session_state:
    st.session_state["cfg"] = {d: dict(c) for d, c in D.DEFAULT_CONFIG.items()}

st.title("FRTB IMA Lab")
st.caption("Internal models approach for market risk: P&L attribution test, desk backtesting, risk factor eligibility "
           "test, non-modellable risk factors, liquidity-adjusted expected shortfall and the aggregate capital "
           "requirement with the PLA surcharge — Basel MAR31–33, CRR3.")

tabs = st.tabs(["Overview", "PLA test", "PLA diagnostics", "Backtesting", "RFET", "ES & IMCC", "NMRF (SES)",
                "Capital", "Import CSV", "Methodology"])

# ---------------------------------------------------------------- diagnostics first: it sets the desk risk models
with tabs[2]:
    st.markdown("The risk model differs from the front-office pricers in ways that real banks know well. Set them per "
                "desk and watch the PLA metrics move; every other tab follows these settings.")
    desk_d = st.selectbox("Desk", list(b.desks), index=5, key="diag_desk")
    cfg = st.session_state["cfg"][desk_d]
    c = st.columns(5)
    cfg["proxy_share"] = c[0].slider("Share of factors proxied", 0.0, 1.0, float(cfg["proxy_share"]), 0.05, key=f"ps_{desk_d}",
                                     help="Factors replaced in the risk model by a correlated proxy (least liquid first).")
    cfg["proxy_corr"] = c[1].slider("Proxy correlation", 0.3, 1.0, float(cfg["proxy_corr"]), 0.05, key=f"pc_{desk_d}")
    cfg["pricer"] = c[2].selectbox("Risk-model pricer", PRICERS, index=PRICERS.index(cfg["pricer"]), key=f"pr_{desk_d}",
                                   help="Full revaluation, delta-gamma (no higher-order terms) or delta only.")
    cfg["missing"] = c[3].slider("Factors missing from the model", 0.0, 1.5, float(cfg["missing"]), 0.05, key=f"mi_{desk_d}",
                                 help="P&L from factors priced in front office but absent from the risk model "
                                      "(basis, dividends, correlation), as a multiple of the modelled P&L volatility.")
    cfg["timing"] = c[4].slider("Data timing mismatch", 0.0, 1.0, float(cfg["timing"]), 0.05, key=f"ti_{desk_d}",
                                help="Noise from snapshot times that differ between front office and risk (e.g. Asian close vs "
                                     "London close), in units of each factor's daily volatility.")
    if st.button("Reset all desks to the default story"):
        st.session_state["cfg"] = {d: dict(c) for d, c in D.DEFAULT_CONFIG.items()}
        for k in list(st.session_state):
            if k[:3] in ("ps_", "pc_", "pr_", "mi_", "ti_"):
                del st.session_state[k]
        st.rerun()

cfgs = st.session_state["cfg"]
cfg_key = tuple((d, tuple(sorted(c.items()))) for d, c in cfgs.items())
pnls = {d: pnl(seed, d, tuple(sorted(cfgs[d].items()))) for d in b.desks}
pla_res = {d: PL.pla_test(p) for d, p in pnls.items()}
bt_res = {d: PL.backtest(p) for d, p in pnls.items()}
rf = RF.run(b.obs, b.factors.index, asof)
mod = rf["modellable"]
status = {}
for d in b.desks:
    z = pla_res[d]["zone"]
    status[d] = z if (z != "red" and bt_res[d]["passes"]) else "SA"
approved = tuple(d for d in b.desks if status[d] != "SA")
mod_key = tuple(mod.items())
nmrf_in = tuple(f for f in rf.index[~mod] if any(f in b.desks[d]["factors"] for d in approved))

with tabs[2]:
    r = pla_res[desk_d]
    c = st.columns(4)
    c[0].metric("Spearman", f"{r['Spearman']:.3f}")
    c[1].metric("Kolmogorov-Smirnov", f"{r['KS']:.3f}")
    c[2].metric("Zone", chip(r["zone"]))
    c[3].metric("Unexplained P&L vol / HPL vol", f"{r['unexplained P&L / HPL vol']:.0%}")
    st.subheader("Which shortcoming costs the most?")
    st.caption("Each shortcoming removed alone, the others kept: the gain in Spearman and KS shows what to fix first.")
    fixes = {"proxies": {"proxy_share": 0.0}, "pricer": {"pricer": "full"}, "missing factors": {"missing": 0.0},
             "timing": {"timing": 0.0}}
    rows = [{"fix": "current model", **{k: r[k] for k in ("Spearman", "KS", "zone")}}]
    for nm, ch in fixes.items():
        rr = PL.pla_test(pnl(seed, desk_d, tuple(sorted({**cfg, **ch}.items()))))
        rows.append({"fix": f"remove {nm}", **{k: rr[k] for k in ("Spearman", "KS", "zone")}})
    rr = PL.pla_test(pnl(seed, desk_d, tuple(sorted({**cfg, "proxy_share": 0.0, "pricer": "full", "missing": 0.0, "timing": 0.0}.items()))))
    rows.append({"fix": "remove all", **{k: rr[k] for k in ("Spearman", "KS", "zone")}})
    fx = pd.DataFrame(rows)
    fx["zone"] = fx["zone"].map(chip)
    st.dataframe(text(fx, {"Spearman": "{:.3f}", "KS": "{:.3f}"}), hide_index=True, width="stretch", height=h(fx))
    st.subheader("Sensitivity to one shortcoming")
    a, bcol = st.columns([1, 3])
    param = a.radio("Vary", ["missing", "proxy_share", "proxy_corr", "timing"],
                    format_func={"missing": "Missing factors", "proxy_share": "Share proxied", "proxy_corr": "Proxy correlation",
                                 "timing": "Timing mismatch"}.get)
    grid = {"missing": np.linspace(0, 1.5, 16), "proxy_share": np.linspace(0, 1, 11), "proxy_corr": np.linspace(0.3, 1, 15),
            "timing": np.linspace(0, 1, 11)}[param]
    sw = pd.DataFrame([{param: v, **{k: PL.pla_test(pnl(seed, desk_d, tuple(sorted({**cfg, param: float(v)}.items()))))[k]
                                     for k in ("Spearman", "KS")}} for v in grid])
    fig = go.Figure()
    fig.add_scatter(x=sw[param], y=sw["Spearman"], name="Spearman", line=dict(color="#7a3e6b"))
    fig.add_scatter(x=sw[param], y=sw["KS"], name="KS", yaxis="y2", line=dict(color="#d99a2b"))
    for y, dash in ((P.PLA["spearman_green"], "dot"), (P.PLA["spearman_red"], "dash")):
        fig.add_hline(y=y, line_dash=dash, line_color="#7a3e6b", opacity=0.5)
    for y, dash in ((P.PLA["ks_green"], "dot"), (P.PLA["ks_red"], "dash")):
        fig.add_hline(y=y, line_dash=dash, line_color="#d99a2b", opacity=0.5, yref="y2")
    fig.update_layout(height=380, margin=dict(t=20, b=10), yaxis=dict(title="Spearman", range=[0, 1.02]),
                      yaxis2=dict(title="KS", overlaying="y", side="right", range=[0, 0.4]), xaxis_title=param,
                      legend=dict(orientation="h", y=-0.2))
    bcol.plotly_chart(fig, width="stretch")
    a.caption("Dotted lines: green thresholds (Spearman 0.80, KS 0.09); dashed: red thresholds (0.70, 0.12).")

# ---------------------------------------------------------------- overview
with tabs[0]:
    im = run_imcc(seed, approved, cfg_key, mod_key) if approved else None
    ses_t = run_ses(seed, approved, cfg_key, nmrf_in, im["stress start"]) if im else pd.DataFrame()
    ses_v = C.aggregate_ses(ses_t)
    bb = C.bank_backtest(pnls, list(approved))
    mult = bb["multiplier"] + extra_plus
    if "sa" not in st.session_state or st.session_state.get("sa_seed") != seed:
        st.session_state["sa"] = default_sa(seed, tuple((d, tuple(sorted(D.DEFAULT_CONFIG[d].items()))) for d in b.desks))
        st.session_state["sa_seed"] = seed
    sa_tab = st.session_state["sa"].copy()
    sa_tab["status"] = pd.Series(status)
    agg = C.aggregate(sa_tab, im["IMCC"] if im else 0.0, ses_v["SES"], mult, avg_ratio)
    c = st.columns(6)
    c[0].metric("Desks on IMA", f"{len(approved)} / {len(b.desks)}")
    c[1].metric("Modellable factors", f"{mod.sum()} / {len(mod)}")
    c[2].metric("IMCC", f"€{agg['IMCC'] / 1e3:,.1f} m")
    c[3].metric("SES (NMRF)", f"€{agg['SES'] / 1e3:,.1f} m")
    c[4].metric("Capital requirement", f"€{agg['ACR'] / 1e3:,.1f} m")
    c[5].metric("vs SA for all desks", f"{agg['ACR'] / agg['SA all desks'] - 1:+.0%}")
    rows = []
    for d in b.desks:
        fs = b.desks[d]["factors"]
        rows.append({"desk": d, "Spearman": pla_res[d]["Spearman"], "KS": pla_res[d]["KS"], "PLA zone": chip(pla_res[d]["zone"]),
                     "exceptions 99 %": bt_res[d]["exc99"], "exceptions 97.5 %": bt_res[d]["exc975"],
                     "backtesting": "pass" if bt_res[d]["passes"] else "fail",
                     "NMRF": int((~mod[fs]).sum()), "factors": len(fs), "approach": chip(status[d]) if status[d] == "SA" else "IMA – " + chip(status[d]),
                     "SA (k€)": sa_tab.loc[d, "SA"], "DRC (k€)": sa_tab.loc[d, "DRC"]})
    ov = pd.DataFrame(rows).set_index("desk")
    st.dataframe(text(ov, {"Spearman": "{:.3f}", "KS": "{:.3f}", "SA (k€)": "{:,.0f}", "DRC (k€)": "{:,.0f}"}),
                 width="stretch", height=h(ov))
    a, bcol = st.columns([3, 2])
    fig = go.Figure()
    xs, ys = np.linspace(0, 0.3, 2), [0.4, 1.0]
    fig.add_shape(type="rect", x0=0, x1=0.3, y0=0.4, y1=1.0, fillcolor=ZC["amber"], opacity=0.10, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=P.PLA["ks_green"], y0=P.PLA["spearman_green"], y1=1.0, fillcolor=ZC["green"], opacity=0.18, line_width=0)
    fig.add_shape(type="rect", x0=P.PLA["ks_red"], x1=0.3, y0=0.4, y1=1.0, fillcolor=ZC["red"], opacity=0.12, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=P.PLA["ks_red"], y0=0.4, y1=P.PLA["spearman_red"], fillcolor=ZC["red"], opacity=0.12, line_width=0)
    for i, d in enumerate(b.desks):
        z = pla_res[d]["zone"]
        fig.add_scatter(x=[pla_res[d]["KS"]], y=[pla_res[d]["Spearman"]], mode="markers+text", text=[d],
                        textposition=["top center", "middle right", "middle left", "bottom center", "middle right", "bottom center"][i % 6],
                        marker=dict(size=13, color=ZC[z], line=dict(color="white", width=1)), showlegend=False)
    fig.update_layout(title="PLA zone map (last 250 days)", height=420, margin=dict(t=40, b=10),
                      xaxis=dict(title="Kolmogorov-Smirnov distance", range=[0, 0.3]),
                      yaxis=dict(title="Spearman correlation", range=[0.4, 1.0]))
    a.plotly_chart(fig, width="stretch")
    avg_branch = mult * agg["IMCC"] * avg_ratio + agg["SES"] * avg_ratio >= agg["IMCC"] + agg["SES"]
    imcc_term = mult * agg["IMCC"] * avg_ratio if avg_branch else agg["IMCC"]
    wf = [("IMCC × m_c" if avg_branch else "IMCC", imcc_term), ("SES", agg["C_A"] - imcc_term),
          ("DRC (IMA desks)", agg["DRC (G, A)"]), ("PLA surcharge", agg["PLA surcharge"]), ("SA of other desks", agg["C_U"])]
    capped = agg["ACR"] - sum(v for _, v in wf)
    if abs(capped) > 1:
        wf.append(("SA cap / floor adjustment", capped))
    fig = go.Figure(go.Waterfall(x=[k for k, _ in wf] + ["Capital requirement"], y=[v / 1e3 for _, v in wf] + [agg["ACR"] / 1e3],
                                 measure=["relative"] * len(wf) + ["total"], connector=dict(line=dict(color="#bbb")),
                                 increasing=dict(marker_color="#7a3e6b"), decreasing=dict(marker_color="#3b8c5a"),
                                 totals=dict(marker_color="#3a3a4a")))
    fig.add_hline(y=agg["SA all desks"] / 1e3, line_dash="dot", annotation_text="SA, all desks")
    fig.update_layout(title="Capital build-up (EUR m)", height=420, margin=dict(t=40, b=10), yaxis_title="EUR m")
    bcol.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- PLA test
with tabs[1]:
    desk = st.selectbox("Desk", list(b.desks), key="pla_desk")
    p = pnls[desk]
    r = pla_res[desk]
    c = st.columns(4)
    c[0].metric("Spearman (≥ 0.80 green, < 0.70 red)", f"{r['Spearman']:.3f}")
    c[1].metric("KS (≤ 0.09 green, > 0.12 red)", f"{r['KS']:.3f}")
    c[2].metric("Zone", chip(r["zone"]))
    c[3].metric("Approach", "IMA" if status[desk] != "SA" else "Standardised")
    w = p.iloc[-250:]
    a, bcol = st.columns(2)
    fig = px.scatter(w, x="RTPL", y="HPL", opacity=0.6, color_discrete_sequence=["#7a3e6b"], title="HPL vs RTPL, last 250 days (k€)")
    lim = float(np.abs(w[["HPL", "RTPL"]]).max().max())
    fig.add_scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", line=dict(color="#999", dash="dot"), showlegend=False)
    fig.update_layout(height=400, margin=dict(t=40, b=10))
    a.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for k, col in (("HPL", "#3a3a4a"), ("RTPL", "#7a3e6b")):
        x = np.sort(w[k])
        fig.add_scatter(x=x, y=np.arange(1, len(x) + 1) / len(x), name=k, line=dict(color=col, shape="hv"))
    fig.update_layout(title=f"Empirical distributions — KS distance {r['KS']:.3f}", height=400, margin=dict(t=40, b=10),
                      xaxis_title="daily P&L (k€)", yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.2))
    bcol.plotly_chart(fig, width="stretch")
    roll = PL.rolling_pla(p)
    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll["Spearman"], name="Spearman", line=dict(color="#7a3e6b"))
    fig.add_scatter(x=roll.index, y=roll["KS"], name="KS", yaxis="y2", line=dict(color="#d99a2b"))
    fig.add_hline(y=P.PLA["spearman_green"], line_dash="dot", line_color="#7a3e6b", opacity=0.5)
    fig.add_hline(y=P.PLA["spearman_red"], line_dash="dash", line_color="#7a3e6b", opacity=0.5)
    fig.add_hline(y=P.PLA["ks_green"], line_dash="dot", line_color="#d99a2b", opacity=0.5, yref="y2")
    fig.add_hline(y=P.PLA["ks_red"], line_dash="dash", line_color="#d99a2b", opacity=0.5, yref="y2")
    fig.update_layout(title="Rolling PLA metrics (250-day windows, quarterly in practice)", height=360, margin=dict(t=40, b=10),
                      yaxis=dict(title="Spearman", range=[0, 1.02]), yaxis2=dict(title="KS", overlaying="y", side="right", range=[0, 0.4]),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("The test is run quarterly on the last 250 days (MAR32.42–32.43). A desk in the red zone moves to the standardised "
               "approach; an amber desk stays on the internal model but triggers the capital surcharge.")

# ---------------------------------------------------------------- backtesting
with tabs[3]:
    desk = st.selectbox("Desk", list(b.desks), key="bt_desk", index=5)
    bt = bt_res[desk]
    p = pnls[desk].iloc[-250:]
    c = st.columns(5)
    c[0].metric("Exceptions 99 % (APL / HPL)", f"{bt['exceptions 99.0% APL']} / {bt['exceptions 99.0% HPL']}", help="Limit 12")
    c[1].metric("Exceptions 97.5 % (APL / HPL)", f"{bt['exceptions 97.5% APL']} / {bt['exceptions 97.5% HPL']}", help="Limit 30")
    c[2].metric("Desk backtesting", "pass" if bt["passes"] else "fail")
    c[3].metric("Bank-wide 99 % exceptions", f"{bb['exceptions']}")
    c[4].metric("Multiplier m_c", f"{mult:.2f}")
    fig = go.Figure()
    fig.add_bar(x=p.index, y=p["APL"], name="actual P&L", marker_color="#9aa5b1")
    fig.add_scatter(x=p.index, y=p["HPL"], name="hypothetical P&L", mode="markers", marker=dict(size=4, color="#3a3a4a"))
    fig.add_scatter(x=p.index, y=-bt["VaR 99.0%"], name="−VaR 99 %", line=dict(color="#b8433a"))
    fig.add_scatter(x=p.index, y=-bt["VaR 97.5%"], name="−VaR 97.5 %", line=dict(color="#d99a2b", dash="dot"))
    exc = p["APL"] < -bt["VaR 99.0%"]
    fig.add_scatter(x=p.index[exc], y=p["APL"][exc], name="exception 99 % (APL)", mode="markers",
                    marker=dict(symbol="x", size=10, color="#b8433a"))
    fig.update_layout(height=420, margin=dict(t=20, b=10), yaxis_title="k€", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    tb = pd.DataFrame([{"desk": d, "99 % APL": bt_res[d]["exceptions 99.0% APL"], "99 % HPL": bt_res[d]["exceptions 99.0% HPL"],
                        "97.5 % APL": bt_res[d]["exceptions 97.5% APL"], "97.5 % HPL": bt_res[d]["exceptions 97.5% HPL"],
                        "result": "pass" if bt_res[d]["passes"] else "fail"} for d in b.desks]).set_index("desk")
    st.dataframe(tb, width="stretch", height=h(tb))
    st.caption("Desk level (MAR32.19): more than 12 exceptions at 99 % or 30 at 97.5 % over 250 days, on actual or "
               "hypothetical P&L, move the desk to the standardised approach. Bank level (MAR33.42): the multiplier is "
               "1.5 plus a plus factor of 0 to 0.5 from the 99 % exceptions of the desks on the internal model.")

# ---------------------------------------------------------------- RFET
with tabs[4]:
    rft = rf.join(b.factors[["risk class", "category", "liquidity horizon", "profile"]])
    c = st.columns(4)
    c[0].metric("Risk factors", len(rft))
    c[1].metric("Modellable", f"{rft['modellable'].sum()}")
    c[2].metric("Pass with ≥ 100 observations", f"{rft['100 obs'].sum()}")
    c[3].metric("Pass with 24 obs + 4 per 90 days only", f"{(rft['24 obs + 4 per 90 days'] & ~rft['100 obs']).sum()}")
    f_sel = st.selectbox("Factor", list(rft.sort_values(["modellable", "observations (12 m)"]).index), key="rf_factor")
    d_obs = pd.DatetimeIndex(b.obs.loc[b.obs["factor"] == f_sel, "date"])
    d_obs = d_obs[d_obs > asof - pd.Timedelta(days=365)]
    rr = rft.loc[f_sel]
    fig = go.Figure()
    ws = rr["weakest window start"]
    fig.add_vrect(x0=ws, x1=ws + pd.Timedelta(days=89), fillcolor="#b8433a" if rr["min obs in 90 days"] < 4 else "#3b8c5a",
                  opacity=0.12, line_width=0, annotation_text=f"weakest 90 days: {rr['min obs in 90 days']} obs.")
    fig.add_scatter(x=d_obs, y=np.ones(len(d_obs)), mode="markers", marker=dict(symbol="line-ns-open", size=22, color="#7a3e6b"),
                    showlegend=False)
    fig.update_layout(height=200, margin=dict(t=30, b=10), yaxis=dict(visible=False), title=
                      f"{f_sel}: {rr['observations (12 m)']} real-price observations in 12 months — "
                      f"{'modellable' if rr['modellable'] else 'non-modellable'}",
                      xaxis=dict(range=[asof - pd.Timedelta(days=365), asof]))
    st.plotly_chart(fig, width="stretch")
    show = rft[["risk class", "category", "liquidity horizon", "observations (12 m)", "min obs in 90 days",
                "24 obs + 4 per 90 days", "100 obs", "modellable"]].sort_values(["modellable", "observations (12 m)"])
    st.dataframe(show, width="stretch", height=min(h(show), 560))
    st.caption("MAR31.12: at least 24 real-price observations over the previous 12 months with no 90-day period containing "
               "fewer than 4, or at least 100 observations. Real prices include trades, committed quotes and vendor data "
               "meeting the criteria of MAR31.13–31.14. The bucketing approach for curves and surfaces is not modelled.")

# ---------------------------------------------------------------- ES & IMCC
with tabs[5]:
    if not im:
        st.warning("No desk is on the internal model.")
    else:
        c = st.columns(5)
        c[0].metric("IMCC", f"€{im['IMCC'] / 1e3:,.1f} m")
        c[1].metric("ES unconstrained", f"€{im['unconstrained']['ES'] / 1e3:,.1f} m")
        c[2].metric("Σ ES by risk class", f"€{im['constrained sum'] / 1e3:,.1f} m")
        c[3].metric("Stress period", f"{im['stress window'][0]:%b %y} – {im['stress window'][1]:%b %y}")
        c[4].metric("Reduced set coverage (≥ 75 %)", f"{im['reduced-set coverage']:.0%}")
        a, bcol = st.columns(2)
        sc = im["scan"]
        fig = px.line(sc, y="ES reduced set", title="Reduced-set ES over rolling 250-day windows (stress period search)",
                      color_discrete_sequence=["#7a3e6b"])
        fig.add_vline(x=im["stress window"][1], line_dash="dot", annotation_text="stress window end")
        fig.update_layout(height=360, margin=dict(t=40, b=10), yaxis_title="k€", xaxis_title="")
        a.plotly_chart(fig, width="stretch")
        parts = im["unconstrained"]["parts"]
        Hh = P.LIQUIDITY_HORIZONS
        scaled = [parts[0]] + [parts[j] * np.sqrt((Hh[j] - Hh[j - 1]) / 10) for j in range(1, 5)]
        fig = go.Figure()
        fig.add_bar(x=[f"LH ≥ {x}" for x in Hh], y=parts, name="ES_T(P, j), 10-day", marker_color="#c9b3c4")
        fig.add_bar(x=[f"LH ≥ {x}" for x in Hh], y=scaled, name="× √((LH_j − LH_j−1)/10)", marker_color="#7a3e6b")
        fig.update_layout(barmode="group", title="Liquidity-horizon cascade (reduced set, stress period)", height=360,
                          margin=dict(t=40, b=10), yaxis_title="k€", legend=dict(orientation="h", y=-0.2))
        bcol.plotly_chart(fig, width="stretch")
        rows = [{"scope": "All classes (unconstrained)", **{k: im["unconstrained"][k] for k in ("ES_F,C", "ES_R,C", "ES_R,S", "ratio", "ES")}}]
        rows += [{"scope": k, **{kk: v[kk] for kk in ("ES_F,C", "ES_R,C", "ES_R,S", "ratio", "ES")}} for k, v in im["by class"].items()]
        et = pd.DataFrame(rows).set_index("scope")
        st.dataframe(text(et, {"ES_F,C": "{:,.0f}", "ES_R,C": "{:,.0f}", "ES_R,S": "{:,.0f}", "ratio": "{:.2f}", "ES": "{:,.0f}"}),
                     width="stretch", height=h(et))
        st.caption(f"ES = ES_R,S × max(ES_F,C / ES_R,C, 1). IMCC = 0.5 × {im['unconstrained']['ES']:,.0f} + 0.5 × "
                   f"{im['constrained sum']:,.0f} = {im['IMCC']:,.0f} k€. Desks on the internal model: {', '.join(approved)}.")

# ---------------------------------------------------------------- SES
with tabs[6]:
    if ses_t.empty:
        st.info("No non-modellable factor on the desks of the internal model.")
    else:
        c = st.columns(4)
        c[0].metric("SES", f"€{ses_v['SES'] / 1e3:,.1f} m")
        c[1].metric("Sum of stand-alone stress losses", f"€{ses_v['sum of stand-alone'] / 1e3:,.1f} m")
        c[2].metric("Idiosyncratic credit (zero correlation)", f"€{ses_v['idio credit'] / 1e3:,.1f} m")
        c[3].metric("Other NMRF (ρ = 0.6)", f"€{ses_v['other'] / 1e3:,.1f} m")
        t = ses_t.sort_values("stress loss", ascending=True)
        fig = px.bar(t, x="stress loss", y=t.index, color="type", orientation="h", title="Stress loss per non-modellable factor (k€)",
                     color_discrete_map={"idiosyncratic credit": "#d99a2b", "idiosyncratic equity": "#3b8c5a", "other": "#7a3e6b"})
        fig.update_layout(height=max(320, 28 * len(t)), margin=dict(t=40, b=10), yaxis_title="", legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(text(ses_t.sort_values("stress loss", ascending=False), {"stress loss": "{:,.0f}"}), width="stretch", height=h(ses_t))
        st.caption("Each NMRF is shocked alone over the stress period; its loss is the 97.5 % expected shortfall of the "
                   "resulting P&L, scaled to its liquidity horizon floored at 20 days (MAR33.16–33.17). Idiosyncratic credit "
                   "and equity NMRFs aggregate with zero correlation; the others with ρ = 0.6.")

# ---------------------------------------------------------------- capital
with tabs[7]:
    st.markdown("**Standardised and default risk charges per desk** (k€) — illustrative values; replace them with the output "
                "of the SA engine (e.g. the companion `frtb-sa-calculator`).")
    ed = st.data_editor(st.session_state["sa"], width="stretch", height=h(st.session_state["sa"]), key=f"sa_ed_{seed}")
    sa_tab = ed.copy()
    sa_tab["status"] = pd.Series(status)
    agg = C.aggregate(sa_tab, im["IMCC"] if im else 0.0, ses_v["SES"], mult, avg_ratio)
    rows = [("IMCC (t−1)", agg["IMCC"]), ("SES (t−1)", agg["SES"]), ("Multiplier m_c", agg["multiplier"]),
            ("C_A = max(IMCC + SES, m_c × IMCC_avg + SES_avg)", agg["C_A"]), ("DRC of IMA desks", agg["DRC (G, A)"]),
            ("IMA_G,A = C_A + DRC", agg["IMA_G,A"]), ("SA_G,A (SA of the same desks)", agg["SA_G,A"]),
            ("k = 0.5 × SA_amber / SA_G,A", agg["k"]), ("PLA surcharge = k × max(0, SA_G,A − IMA_G,A)", agg["PLA surcharge"]),
            ("C_U (SA of desks out of the model)", agg["C_U"]), ("SA for all desks", agg["SA all desks"]),
            ("ACR = min(IMA_G,A + surcharge + C_U, SA_all) + max(0, IMA_G,A − SA_G,A)", agg["ACR"])]
    ct = pd.DataFrame(rows, columns=["item", "value"])
    ct["value"] = [f"{v:.2f}" if i in ("Multiplier m_c", "k = 0.5 × SA_amber / SA_G,A") else f"{v:,.0f} k€"
                   for i, v in zip(ct["item"], ct["value"])]
    a, bcol = st.columns([3, 2])
    a.dataframe(ct, hide_index=True, width="stretch", height=h(ct))
    with bcol:
        st.metric("Capital requirement", f"€{agg['ACR'] / 1e3:,.1f} m", f"{agg['ACR'] / agg['SA all desks'] - 1:+.1%} vs SA for all desks",
                  delta_color="inverse")
        st.markdown(f"* Binding: **{agg['binding']}**.\n* Amber desks: "
                    f"{', '.join(d for d in b.desks if status[d] == 'amber') or 'none'}.\n* Desks on SA: "
                    f"{', '.join(d for d in b.desks if status[d] == 'SA') or 'none'}.")
    st.subheader("What if…")
    scen = {"Current": status,
            "All IMA desks green": {d: ("green" if s != "SA" else s) for d, s in status.items()},
            "Red desks fixed (green)": {d: ("green" if s == "SA" else s) for d, s in status.items()},
            "Every desk on SA": {d: "SA" for d in status}}
    wi = []
    for k, stt in scen.items():
        tt = ed.copy()
        tt["status"] = pd.Series(stt)
        ap = tuple(d for d in b.desks if stt[d] != "SA")
        imx = run_imcc(seed, ap, cfg_key, mod_key)["IMCC"] if ap else 0.0
        nm = tuple(f for f in rf.index[~mod] if any(f in b.desks[d]["factors"] for d in ap))
        sx = C.aggregate_ses(run_ses(seed, ap, cfg_key, nm, run_imcc(seed, ap, cfg_key, mod_key)["stress start"]))["SES"] if ap else 0.0
        mx = C.bank_backtest(pnls, list(ap))["multiplier"] + extra_plus
        g = C.aggregate(tt, imx, sx, mx, avg_ratio)
        wi.append({"scenario": k, "IMA desks": len(ap), "IMCC": imx, "SES": sx, "PLA surcharge": g["PLA surcharge"], "C_U": g["C_U"],
                   "capital": g["ACR"], "vs SA all": g["ACR"] / g["SA all desks"] - 1})
    wi = pd.DataFrame(wi).set_index("scenario")
    st.dataframe(text(wi, {"IMCC": "{:,.0f}", "SES": "{:,.0f}", "PLA surcharge": "{:,.0f}", "C_U": "{:,.0f}", "capital": "{:,.0f}",
                           "vs SA all": "{:+.1%}"}), width="stretch", height=h(wi))
    st.caption("“Red desks fixed” keeps the same factor positions and simply counts the desks as green: the IMCC rises "
               "(more positions), but the standardised charge of those desks disappears. In practice the 60-day averages, the "
               "quarterly PLA cycle and the conditions for returning to the internal model after a red zone delay these effects.")

# ---------------------------------------------------------------- import
with tabs[8]:
    a, bcol = st.columns(2)
    with a:
        st.subheader("Desk P&L series → PLA and backtesting")
        st.download_button("Template (current synthetic desks)", IO.pnl_template(pnls).to_csv(index=False).encode(),
                           "pla_pnl_template.csv", "text/csv")
        up = st.file_uploader("CSV: date, desk, HPL, RTPL[, APL]", type="csv", key="up_pnl")
        if up is not None:
            try:
                imp = IO.read_pnl(pd.read_csv(up))
                rows = []
                for d, pp in imp.items():
                    r = PL.pla_test(pp)
                    btx = PL.backtest(pp) if len(pp) >= 500 else None
                    rows.append({"desk": d, "days": len(pp), "Spearman": r["Spearman"], "KS": r["KS"], "zone": chip(r["zone"]),
                                 "exceptions 99 %": btx["exc99"] if btx else np.nan,
                                 "exceptions 97.5 %": btx["exc975"] if btx else np.nan})
                res = pd.DataFrame(rows).set_index("desk")
                st.dataframe(text(res, {"Spearman": "{:.3f}", "KS": "{:.3f}", "exceptions 99 %": "{:.0f}", "exceptions 97.5 %": "{:.0f}"}),
                             width="stretch", height=h(res))
                st.caption("Backtesting needs 500 days (250 for the VaR window, 250 tested).")
            except ValueError as e:
                st.error(str(e))
    with bcol:
        st.subheader("Real-price observations → RFET")
        tmpl = b.obs[b.obs["date"] > asof - pd.Timedelta(days=365)].sort_values(["factor", "date"])
        st.download_button("Template (current synthetic log)", tmpl.to_csv(index=False).encode(), "rfet_observations_template.csv",
                           "text/csv")
        up2 = st.file_uploader("CSV: factor, date", type="csv", key="up_obs")
        asof2 = st.date_input("Test date", asof.date())
        if up2 is not None:
            try:
                ob = IO.read_obs(pd.read_csv(up2))
                res = RF.run(ob, pd.Index(sorted(ob["factor"].unique())), pd.Timestamp(asof2))
                st.metric("Modellable", f"{res['modellable'].sum()} / {len(res)}")
                st.dataframe(res.drop(columns="weakest window start"), width="stretch", height=min(h(res), 520))
            except ValueError as e:
                st.error(str(e))

# ---------------------------------------------------------------- methodology
with tabs[9]:
    st.markdown(r"""
### Synthetic bank
Six desks (rates, FX options, credit, equity derivatives, commodities, structured products) hold non-linear positions
(delta, gamma, third-order terms) in 48 risk factors. The factor moves follow a global + risk-class + idiosyncratic
model with Student t innovations over four years, including a stress year. Real-price observations follow liquidity
profiles: liquid, moderate, sparse, gap (a long blackout) and borderline (a thin quarter).

**Three P&L series per desk.**
* **HPL**: the front-office full revaluation on every factor, plus basis factors absent from the risk model.
* **RTPL**: the risk model, with four shortcomings that can be set per desk:
  * proxies replacing illiquid factors;
  * a simplified pricer (delta-gamma or delta);
  * missing factors;
  * a market-data timing mismatch.
* **APL**: HPL plus intraday trading, new deals and fees.

### Tests
* **PLA** (MAR32.38–32.47), on the last 250 days:
  * Spearman correlation between HPL and RTPL, and Kolmogorov-Smirnov distance between their distributions;
  * green: Spearman ≥ 0.80 and KS ≤ 0.09;
  * red: Spearman < 0.70 or KS > 0.12;
  * amber otherwise.

  Red desks go to the standardised approach; amber desks trigger the surcharge.
* **Desk backtesting** (MAR32.19): the 1-day VaR at 99 % and 97.5 % comes from historical simulation on the previous 250 days of RTPL. A desk fails with more than 12 exceptions at 99 % or more than 30 at 97.5 %, on APL or HPL.
* **Bank-wide backtesting** (MAR33.42): the multiplier is 1.5 plus a plus factor of 0 to 0.5 from the 99 % exceptions of the internal-model desks.
* **RFET** (MAR31.12): a factor is modellable with ≥ 24 real-price observations in 12 months and no 90-day period with fewer than 4, or with ≥ 100 observations.

### Capital (MAR33)
$$ES = \sqrt{ES_T(P)^2 + \sum_{j\ge 2}\left(ES_T(P,j)\sqrt{\tfrac{LH_j-LH_{j-1}}{T}}\right)^2},\quad T = 10$$
$$ES = ES_{R,S}\times\max\!\left(\frac{ES_{F,C}}{ES_{R,C}},\,1\right),\qquad IMCC = 0.5\,ES_{\text{all}} + 0.5\sum_{c} ES_c$$
$$SES = \sqrt{\textstyle\sum ISES^2_{\text{idio credit}}} + \sqrt{\textstyle\sum ISES^2_{\text{idio equity}}} + \sqrt{\left(\rho\textstyle\sum SES_k\right)^2 + (1-\rho^2)\textstyle\sum SES_k^2},\ \rho = 0.6$$
$$C_A = \max(IMCC_{t-1} + SES_{t-1},\ m_c\,IMCC_{avg} + SES_{avg}),\qquad IMA_{G,A} = C_A + DRC$$
$$ACR = \min\left(IMA_{G,A} + k\max(0, SA_{G,A} - IMA_{G,A}) + C_U,\ SA_{all}\right) + \max(0, IMA_{G,A} - SA_{G,A}),\quad k = 0.5\,\frac{\sum_{amber} SA_i}{\sum_{G,A} SA_i}$$

### EU context
* **Current framework.** CRR3 (Regulation (EU) 2024/1623) transposes the framework, and the EBA RTS specify the RFET, PLA, backtesting and NMRF requirements.
* **Postponements.** The Commission twice postponed the application of the FRTB capital requirements, to 1 January 2027.
* **Temporary relief.** On 4 June 2026 it adopted a delegated regulation applying for three years from 1 January 2027, subject to scrutiny by Parliament and Council. It temporarily relaxes parts of the IMA, including the PLA test and the RFET, and adds a targeted multiplier.

This lab implements the Basel text; the EU temporary relief is not modelled.

### Limits
* **Simplified model.**
  * Static positions.
  * No DRC model: the DRC and SA charges are inputs.
  * The 60-day averages are proxied by a ratio.
  * No RFET bucketing approach.
  * The NMRF stress scenarios use the factor's own history over the stress period.
* **Not modelled.** The quarterly cycle and the return periods after a red zone.
""")
