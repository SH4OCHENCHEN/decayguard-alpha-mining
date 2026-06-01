from __future__ import annotations

import numpy as np
import pandas as pd

from .ic import summarize_ic


def split_ic(ic_df: pd.DataFrame, train_end: str, valid_end: str) -> dict[str, pd.DataFrame]:
    train = ic_df.loc[:train_end]
    valid = ic_df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1):valid_end]
    test = ic_df.loc[pd.Timestamp(valid_end) + pd.Timedelta(days=1):]
    return {"train": train, "valid": valid, "test": test}


def rolling_decay_slope(ic_df: pd.DataFrame, window: int = 60) -> float:
    s = ic_df["rank_ic"].rolling(window, min_periods=max(20, window // 3)).mean().dropna()
    if len(s) < 5:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    slope = np.polyfit(x, s.to_numpy(), deg=1)[0]
    return float(slope)


def summarize_decay(ic_df: pd.DataFrame, train_end: str, valid_end: str) -> dict[str, float]:
    parts = split_ic(ic_df, train_end, valid_end)
    out: dict[str, float] = {}
    for name, sub in parts.items():
        out.update(summarize_ic(sub, prefix=f"{name}_"))
    out["rank_ic_train_test_gap"] = out.get("train_rank_ic_mean", np.nan) - out.get("test_rank_ic_mean", np.nan)
    out["ic_train_test_gap"] = out.get("train_ic_mean", np.nan) - out.get("test_ic_mean", np.nan)
    out["rolling_rank_ic_slope"] = rolling_decay_slope(ic_df)
    return out
