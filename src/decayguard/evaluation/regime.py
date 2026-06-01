from __future__ import annotations

import numpy as np
import pandas as pd


def make_regimes(close: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    market = returns.mean(axis=1)
    market_ret_60 = market.rolling(60, min_periods=20).sum()
    market_vol_20 = market.rolling(20, min_periods=10).std()
    dispersion = returns.std(axis=1)
    regimes = pd.DataFrame(index=returns.index)
    regimes["bull"] = market_ret_60 > 0
    regimes["bear"] = market_ret_60 <= 0
    regimes["high_vol"] = market_vol_20 > market_vol_20.expanding().median()
    regimes["low_vol"] = ~regimes["high_vol"]
    regimes["high_disp"] = dispersion > dispersion.expanding().median()
    regimes["low_disp"] = ~regimes["high_disp"]
    return regimes.fillna(False)


def regime_summary(ic_df: pd.DataFrame, regimes: pd.DataFrame) -> dict[str, float]:
    common = ic_df.index.intersection(regimes.index)
    out = {}
    vals = []
    for name in regimes.columns:
        mask = regimes.loc[common, name].astype(bool)
        s = ic_df.loc[common[mask], "rank_ic"].dropna()
        val = float(s.mean()) if len(s) else float("nan")
        out[f"regime_{name}_rank_ic"] = val
        if not np.isnan(val):
            vals.append(val)
    out["regime_stability"] = float(np.nanmean(vals) - np.nanstd(vals)) if vals else float("nan")
    return out
