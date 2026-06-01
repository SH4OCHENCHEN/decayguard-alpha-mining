from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ic(factor: pd.DataFrame, future_ret: pd.DataFrame) -> pd.DataFrame:
    idx = factor.index.intersection(future_ret.index)
    cols = factor.columns.intersection(future_ret.columns)
    f = factor.loc[idx, cols]
    y = future_ret.loc[idx, cols]
    rows = []
    for dt in idx:
        x = f.loc[dt]
        r = y.loc[dt]
        mask = x.notna() & r.notna()
        if mask.sum() < 10:
            rows.append((dt, np.nan, np.nan))
            continue
        ic = x[mask].corr(r[mask], method="pearson")
        rank_ic = x[mask].corr(r[mask], method="spearman")
        rows.append((dt, ic, rank_ic))
    return pd.DataFrame(rows, columns=["date", "ic", "rank_ic"]).set_index("date")


def summarize_ic(ic_df: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    out = {}
    for col in ["ic", "rank_ic"]:
        s = ic_df[col].dropna()
        if len(s) == 0:
            out[f"{prefix}{col}_mean"] = np.nan
            out[f"{prefix}{col}_ir"] = np.nan
            out[f"{prefix}{col}_pos_ratio"] = np.nan
        else:
            out[f"{prefix}{col}_mean"] = float(s.mean())
            out[f"{prefix}{col}_ir"] = float(s.mean() / (s.std(ddof=1) + 1e-12))
            out[f"{prefix}{col}_pos_ratio"] = float((s > 0).mean())
    return out
