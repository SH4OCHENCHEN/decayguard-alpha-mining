from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n_days: int = 900, n_assets: int = 80, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Create synthetic wide OHLCV panels: index=date, columns=asset.

    The generator intentionally includes weak momentum/reversal/volatility structures so
    the pipeline can find non-random signals in a demo. Replace this with Qlib or real CSV
    loading for actual experiments.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    assets = [f"S{i:03d}" for i in range(n_assets)]

    market = rng.normal(0.0002, 0.008, size=n_days)
    style = rng.normal(0, 0.004, size=(n_days, 3))
    exposures = rng.normal(0, 1, size=(n_assets, 3))
    eps = rng.normal(0, 0.015, size=(n_days, n_assets))
    ret = market[:, None] + style @ exposures.T * 0.2 + eps

    # Add weak predictable structure: reversal after large negative move and momentum drift.
    for t in range(20, n_days):
        ret[t] += -0.06 * ret[t - 1] + 0.015 * np.mean(ret[t - 20:t], axis=0)

    close = 100 * np.exp(np.cumsum(ret, axis=0))
    close = pd.DataFrame(close, index=dates, columns=assets)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.002, 0.003, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.002, 0.003, close.shape)))
    volume = pd.DataFrame(rng.lognormal(mean=13, sigma=0.5, size=close.shape), index=dates, columns=assets)
    vwap = (open_ + high + low + close) / 4
    return {
        "Open": open_.astype(float),
        "High": high.astype(float),
        "Low": low.astype(float),
        "Close": close.astype(float),
        "Volume": volume.astype(float),
        "VWAP": vwap.astype(float),
    }


def add_derived_fields(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = data["Close"]
    out = dict(data)
    out["Return_1d"] = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def make_future_return(close: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    return close.shift(-horizon) / close - 1.0
