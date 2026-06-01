from __future__ import annotations

import numpy as np
import pandas as pd


def long_short_weights(factor: pd.DataFrame, long_frac: float = 0.1, short_frac: float = 0.1, rebalance_freq: int = 5) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    last_w = pd.Series(0.0, index=factor.columns)
    for i, dt in enumerate(factor.index):
        if i % rebalance_freq == 0:
            x = factor.loc[dt].dropna()
            n = len(x)
            if n >= 20:
                n_long = max(1, int(n * long_frac))
                n_short = max(1, int(n * short_frac))
                w = pd.Series(0.0, index=factor.columns)
                longs = x.nlargest(n_long).index
                shorts = x.nsmallest(n_short).index
                w.loc[longs] = 1.0 / n_long
                w.loc[shorts] = -1.0 / n_short
                last_w = w
        weights.loc[dt] = last_w
    return weights


def portfolio_returns(weights: pd.DataFrame, next_ret: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    idx = weights.index.intersection(next_ret.index)
    cols = weights.columns.intersection(next_ret.columns)
    w = weights.loc[idx, cols].fillna(0.0)
    r = next_ret.loc[idx, cols].fillna(0.0)
    gross = (w.shift(1).fillna(0.0) * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    cost = turnover * cost_bps / 10000.0
    net = gross - cost
    return pd.DataFrame({"gross_ret": gross, "cost": cost, "turnover": turnover, "net_ret": net})


def perf_summary(port: pd.DataFrame, periods_per_year: int = 252) -> dict[str, float]:
    r = port["net_ret"].dropna()
    if len(r) == 0:
        return {}
    equity = (1 + r).cumprod()
    ann_ret = equity.iloc[-1] ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1
    return {
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "turnover": float(port["turnover"].mean()),
        "avg_cost": float(port["cost"].mean()),
    }
