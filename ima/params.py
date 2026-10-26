"""
Regulatory parameters of the FRTB internal models approach (Basel Committee, MAR31–MAR33; transposed in the EU by
CRR3 and the EBA regulatory technical standards on RFET, PLA, backtesting and NMRF).
"""
from __future__ import annotations

# MAR33.12 – liquidity horizons (days) by risk factor category
LIQUIDITY_HORIZONS = [10, 20, 40, 60, 120]
LH_CATEGORY = {
    "GIRR – specified currency": 10, "GIRR – other currency": 20, "GIRR – volatility": 60, "GIRR – other": 60,
    "CSR – sovereign IG": 20, "CSR – sovereign HY": 40, "CSR – corporate IG": 40, "CSR – corporate HY": 60,
    "CSR – volatility": 120, "CSR – other": 120,
    "EQ – large cap": 10, "EQ – small cap": 20, "EQ – large cap volatility": 20, "EQ – small cap volatility": 60,
    "EQ – other": 60,
    "FX – specified pair": 10, "FX – other pair": 20, "FX – volatility": 40, "FX – other": 40,
    "COM – energy & carbon": 20, "COM – precious metals": 20, "COM – other": 60, "COM – energy volatility": 60,
    "COM – other volatility": 120,
}
BASE_HORIZON = 10
ES_ALPHA = 0.975
IMCC_RHO = 0.5            # MAR33.15 – weight of the unconstrained ES
SES_RHO = 0.6             # MAR33.17 – correlation between non-idiosyncratic NMRF stress losses
REDUCED_SET_MIN = 0.75    # MAR33.5 – reduced set must explain at least 75 % of the full-set ES (current period)

# MAR31.12 – risk factor eligibility test
RFET = {"min_obs_with_gap_rule": 24, "window_days": 90, "min_obs_per_window": 4, "min_obs_alone": 100,
        "lookback_days": 365}

# MAR32.42–32.43 – PLA test thresholds (250 observations)
PLA = {"spearman_green": 0.80, "spearman_red": 0.70, "ks_green": 0.09, "ks_red": 0.12, "obs": 250}

# MAR32.19 / 33.43 – backtesting
DESK_BT = {"exc_99_max": 12, "exc_975_max": 30}
# bank-wide backtesting plus factor by number of 99 % exceptions over 250 days (MAR33.42 / MAR99 traffic light)
PLUS_FACTOR = {**{i: 0.0 for i in range(5)}, 5: 0.20, 6: 0.26, 7: 0.33, 8: 0.38, 9: 0.42}
MULTIPLIER_BASE = 1.5


def plus_factor(exceptions: int) -> float:
    return PLUS_FACTOR.get(int(exceptions), 0.5)
