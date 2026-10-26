# FRTB IMA Lab – PLA Test, Risk Factor Eligibility and Non-Modellable Risk Factors

An interactive Streamlit application on the eligibility and capital mechanics of the FRTB **internal models approach (IMA)** for market risk (Basel MAR31–33, CRR3). It covers:

* the **P&L attribution (PLA) test**: Spearman correlation and Kolmogorov-Smirnov distance between hypothetical and risk-theoretical P&L, with green, amber and red zones;
* **PLA diagnostics**: the four classic causes of failure, set per desk, with their individual cost;
* **desk and bank-wide backtesting**: VaR at 99 % and 97.5 %, and the multiplier;
* the **risk factor eligibility test (RFET)**: 24 observations with at least 4 per 90 days, or 100 observations;
* **non-modellable risk factors**: stress scenarios and **SES** aggregation;
* **liquidity-adjusted expected shortfall** with the reduced-set stress calibration and the **IMCC**;
* the **aggregate capital requirement**, with the **PLA surcharge**, the SA cap and what-if scenarios;
* **CSV import** of desk P&L series and real-price observation logs.

**Live app:** `https://<your-app-name>.streamlit.app` *(to be completed after deployment)*

![tests](https://github.com/ldenarbonne/frtb-ima-lab/actions/workflows/tests.yml/badge.svg)

Companion repository for the standardised approach: [`frtb-sa-calculator`](https://github.com/ldenarbonne/frtb-sa-calculator).

---

## 1. Features

| Tab | Content |
|---|---|
| **Overview** | Status of each desk: PLA zone, backtesting, NMRF count, approach. PLA zone map. Capital build-up from IMCC to the requirement |
| **PLA test** | HPL vs RTPL scatter, empirical distributions (KS), rolling Spearman and KS against the thresholds |
| **PLA diagnostics** | Per-desk risk-model settings: share of proxied factors and proxy correlation; pricer (full, delta-gamma, delta); factors missing from the model; data timing mismatch. Shows the gain from removing each one, and sensitivity curves |
| **Backtesting** | VaR 99 % and 97.5 % against APL and HPL, exceptions per desk, bank-wide plus factor and multiplier |
| **RFET** | Observations over 12 months, weakest 90-day window, the two criteria, observation timeline per factor |
| **ES & IMCC** | Stress-period search, liquidity-horizon cascade, ES_F,C / ES_R,C / ES_R,S by risk class, reduced-set coverage, IMCC |
| **NMRF (SES)** | Stress loss per NMRF and aggregation (idiosyncratic credit / equity vs others with ρ = 0.6) |
| **Capital** | Editable SA and DRC per desk; C_A, IMA_G,A, k, PLA surcharge, C_U and the aggregate requirement; what-if scenarios |
| **Import CSV** | Desk P&L (`date, desk, HPL, RTPL[, APL]`) and observations (`factor, date`), with templates |
| **Methodology** | Formulas, EU context, limits |

The PLA settings of each desk are kept for the session and feed every tab, so a change to a desk's risk model moves its zone, the desks on the internal model, the IMCC, the SES and the capital.

## 2. Synthetic bank

* **Desks and risk factors.** Six desks (rates G10, FX options, credit IG, equity derivatives, commodities, structured products) hold non-linear positions in **48 risk factors**. Each factor has its liquidity-horizon category (MAR33.12) and a liquidity profile: liquid, moderate, sparse, gap or borderline.
* **Market history.** Four years of daily moves (global + risk-class + idiosyncratic factors, Student t) include a stress year. Real-price observation logs follow each factor's liquidity profile.
* **Three P&L series per desk:**
  * **HPL**: front-office full revaluation, including basis factors that the risk model does not have;
  * **RTPL**: the risk model, with its shortcomings;
  * **APL**: HPL plus intraday trading and fees.
* **Default story** (seed 7, as of 30 September 2026):

| Desk | Risk-model shortcoming | Spearman | KS | Zone | Exceptions 99 % | Approach |
|---|---|---|---|---|---|---|
| Rates G10 | almost none | 0.994 | 0.028 | green | 5 | IMA |
| FX options | Asian / London snapshot timing | 0.851 | 0.088 | green | 2 | IMA |
| Credit IG | proxies for illiquid names, missing basis | 0.789 | 0.084 | amber | 1 | IMA + surcharge |
| Equity derivatives | delta-gamma pricer, missing dividend / repo factors | 0.784 | 0.108 | amber | 8 | IMA + surcharge |
| Commodities | some proxies, timing | 0.938 | 0.048 | green | 3 | IMA |
| Structured products | delta pricer, proxies, missing correlation / long-dated vol | 0.665 | 0.108 | red | 15 (fails) | SA |

## 3. Illustrative results

* **RFET.** 32 of 48 factors are modellable: 23 pass with 100 observations or more, and 9 only through the 24 + 4-per-90-days rule. The 16 NMRFs are the sparse factors, those with a gap, and some borderline or moderate factors with a thin quarter.
* **IMCC.** €37.8 m: an unconstrained ES of €22.4 m and a sum of ES by risk class of €53.2 m.
* **SES.** €11.3 m, against €16.7 m of stand-alone stress losses.
* **Capital requirement.** €108.6 m, against €125.2 m with the standardised approach for every desk (−13 %). It includes a PLA surcharge of €5.0 m due to the two amber desks.

| What if… | PLA surcharge | SA of desks out of the model | Capital |
|---|---|---|---|
| Current | €5.0 m | €27.8 m | €108.6 m |
| Amber desks back to green | – | €27.8 m | €103.5 m |
| Structured products back on the model | €6.3 m | – | €96.6 m |

**Reading.**

* **Missing factors are usually the first cause of failure.** For structured products, removing the missing factors alone takes Spearman from 0.67 to 0.97. Removing the proxies or the simplified pricer changes little.
* **Timing mismatches fail through the KS test.** They inflate the variance of RTPL more than they break the ranking of HPL and RTPL, so they move the KS distance more than the Spearman correlation.
* **An amber zone costs capital.** The surcharge is k × (SA − IMA) of the eligible desks, with k = 0.5 × the amber share of their SA.
* **A red desk costs more.** It falls back entirely to the standardised approach.
* **NMRFs are a large add-on.** The SES adds about 30 % to the IMCC. It is driven by long liquidity horizons (volatility, dividends, correlation) because each stress loss is scaled by √(LH / 10).

## 4. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests (11 tests, about 15 seconds) cover:

* PLA zones;
* a perfect model gives a green zone;
* PLA metrics deteriorate monotonically with each shortcoming;
* the default story;
* the exception count;
* the RFET rules (daily, regular, gap and burst series);
* ES and the liquidity-horizon cascade;
* the IMCC and SES identities;
* the aggregate capital formula;
* the CSV round trip;
* the app.

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 5. Repository structure

```
app.py           Streamlit application (10 tabs)
ima/params.py    Liquidity horizons, PLA thresholds, RFET criteria, backtesting limits, plus factors
ima/data.py      Synthetic factors, moves, observation logs, desks, HPL / RTPL / APL with risk-model shortcomings
ima/pla.py       Spearman, KS, zones, rolling PLA, VaR backtesting, sensitivity sweeps
ima/rfet.py      Risk factor eligibility test
ima/capital.py   Liquidity-adjusted ES, stress period, IMCC, SES, bank backtesting, aggregate capital
ima/io.py        CSV import and templates
tests/           pytest suite
```

## 6. Regulatory references

* Basel Committee:
  * MAR31 – internal models approach: model requirements (RFET MAR31.12–31.14);
  * [MAR32](https://www.sif.admin.ch/dam/sif/de/dokumente/basel-3/MAR32.pdf.download.pdf/MAR32.pdf) – backtesting and PLA test requirements (desk backtesting MAR32.19, PLA zones MAR32.42–32.43);
  * MAR33 – capital requirements calculation.
* Regulation (EU) 2024/1623 (CRR3) and the EBA regulatory technical standards on RFET, PLA, backtesting and NMRF stress scenarios.
* European Commission, [Delegated Regulation postponing the FRTB](https://www.regulationtomorrow.com/2025/09/commission-delegated-regulation-postponing-application-of-frtb/) (application on 1 January 2027).
* European Commission, [temporary amendments to the market risk rules](https://finance.ec.europa.eu/news/eu-temporarily-amends-prudential-rules-banks-market-risk-2026-06-08_en). Adopted on 4 June 2026, they apply for three years from 1 January 2027 and include a targeted multiplier and a temporary relaxation of the PLA test and the RFET.

The lab implements the Basel text; the EU temporary relief is not modelled.

## 7. Author

**Laurent Motais de Narbonne** – Risk management and quantitative finance · [ldenarbonne.github.io](https://ldenarbonne.github.io/)

Licence: MIT.
