from __future__ import annotations

import numpy as np
import pandas as pd

from .parser import Node

EPS = 1e-12


def _cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def _ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=max(2, window // 2)).apply(
        lambda a: pd.Series(a).rank(pct=True).iloc[-1], raw=False
    )


def _decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return x.rolling(window, min_periods=max(2, window // 2)).apply(lambda a: float(np.dot(a, weights[-len(a):])), raw=True)


def evaluate_node(node: Node, data: dict[str, pd.DataFrame]) -> pd.DataFrame | float | int:
    if node.is_leaf():
        if isinstance(node.value, str):
            return data[node.value]
        return node.value

    op = node.op
    vals = [evaluate_node(a, data) for a in node.args]

    if op == "Add": return vals[0] + vals[1]
    if op == "Sub": return vals[0] - vals[1]
    if op == "Mul": return vals[0] * vals[1]
    if op == "Div": return vals[0] / (vals[1] + EPS)
    if op == "Neg": return -vals[0]
    if op == "Abs": return vals[0].abs()
    if op == "Log": return np.log(vals[0].abs() + EPS)
    if op == "Min": return np.minimum(vals[0], vals[1])
    if op == "Max": return np.maximum(vals[0], vals[1])
    if op == "Rank": return _cs_rank(vals[0])
    if op == "ZScore": return _cs_zscore(vals[0])
    if op == "TsMean": return vals[0].rolling(int(vals[1]), min_periods=max(2, int(vals[1]) // 2)).mean()
    if op == "TsStd": return vals[0].rolling(int(vals[1]), min_periods=max(2, int(vals[1]) // 2)).std()
    if op == "TsRank": return _ts_rank(vals[0], int(vals[1]))
    if op == "Delta": return vals[0] - vals[0].shift(int(vals[1]))
    if op == "Corr": return vals[0].rolling(int(vals[2]), min_periods=max(2, int(vals[2]) // 2)).corr(vals[1])
    if op == "DecayLinear": return _decay_linear(vals[0], int(vals[1]))
    raise ValueError(f"Unsupported operator: {op}")


def clean_factor(factor: pd.DataFrame) -> pd.DataFrame:
    factor = factor.replace([np.inf, -np.inf], np.nan)
    # Cross-sectional winsorization by date.
    lo = factor.quantile(0.01, axis=1)
    hi = factor.quantile(0.99, axis=1)
    factor = factor.clip(lower=lo, upper=hi, axis=0)
    return _cs_zscore(factor).replace([np.inf, -np.inf], np.nan)
