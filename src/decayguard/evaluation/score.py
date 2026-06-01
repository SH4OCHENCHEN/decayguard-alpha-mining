from __future__ import annotations

import math


def safe(x: float, default: float = 0.0) -> float:
    return default if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)


def decayguard_score(metrics: dict, weights: dict | None = None) -> float:
    w = weights or {}
    score = 0.0
    score += w.get("valid_rank_ic", 0.5) * safe(metrics.get("valid_rank_ic_mean"))
    score += w.get("test_rank_ic", 1.0) * safe(metrics.get("test_rank_ic_mean"))
    score += w.get("train_test_gap", -0.7) * abs(safe(metrics.get("rank_ic_train_test_gap")))
    score += w.get("turnover", -0.2) * safe(metrics.get("turnover"))
    score += w.get("complexity", -0.03) * safe(metrics.get("complexity"))
    score += w.get("max_corr", -0.3) * safe(metrics.get("max_corr"))
    slope = safe(metrics.get("rolling_rank_ic_slope"))
    score += w.get("negative_decay_slope", -0.5) * max(0.0, -slope)
    score += w.get("regime_stability", 0.2) * safe(metrics.get("regime_stability"))
    return float(score)
