#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from decayguard.backtest.portfolio import long_short_weights, portfolio_returns, perf_summary
from decayguard.evaluation.decay import summarize_decay
from decayguard.evaluation.ic import daily_ic
from decayguard.evaluation.redundancy import max_factor_corr
from decayguard.evaluation.regime import make_regimes, regime_summary
from decayguard.evaluation.score import decayguard_score
from decayguard.factors.calculator import AlphaCandidate, calculate_factor
from decayguard.factors.data import add_derived_fields, make_future_return, make_synthetic_ohlcv

FALLBACK = [
    {"hypothesis": "Short-term reversal after sharp price drops.", "formula": "Rank(Neg(Delta(Close, 5)))", "rationale": "Recent losers may mean-revert.", "expected_regime": "high_vol", "failure_mode": "strong momentum market", "expected_turnover": "medium"},
    {"hypothesis": "High recent volatility amplifies reversal.", "formula": "Mul(Rank(Neg(Delta(Close, 5))), Rank(TsStd(Return_1d, 20)))", "rationale": "Liquidity shocks create temporary dislocations.", "expected_regime": "high_vol", "failure_mode": "persistent crash", "expected_turnover": "medium"},
    {"hypothesis": "Sustained trend continuation.", "formula": "Rank(Delta(TsMean(Close, 5), 20))", "rationale": "Smoothed price trend may persist.", "expected_regime": "bull", "failure_mode": "choppy market", "expected_turnover": "low"},
    {"hypothesis": "Volume-price divergence indicates pressure.", "formula": "Neg(Corr(Return_1d, Volume, 20))", "rationale": "Rising volume on weak returns can signal distribution.", "expected_regime": "bear", "failure_mode": "news-driven volume", "expected_turnover": "medium"},
    {"hypothesis": "Low-vol stocks with positive drift outperform.", "formula": "Mul(Rank(Delta(Close, 20)), Rank(Neg(TsStd(Return_1d, 20))))", "rationale": "Quality-like trend with lower noise.", "expected_regime": "low_vol", "failure_mode": "risk-on reversal", "expected_turnover": "low"},
]


def load_candidates(path: Path | None) -> list[dict]:
    if path and path.exists():
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return FALLBACK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--candidates", default="results/candidates.jsonl")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    data = add_derived_fields(make_synthetic_ohlcv(cfg["n_days"], cfg["n_assets"], cfg["seed"]))
    future_ret = make_future_return(data["Close"], cfg["label_horizon"])
    next_ret = data["Close"].pct_change().shift(-1)
    regimes = make_regimes(data["Close"], data["Return_1d"])

    existing_factors = []
    rows = []
    factors = {}
    candidates = load_candidates(Path(args.candidates))

    for item in candidates:
        cand = AlphaCandidate(**{k: item.get(k, "") for k in AlphaCandidate.__dataclass_fields__.keys()})
        try:
            ef = calculate_factor(cand, data)
            ic = daily_ic(ef.factor, future_ret)
            metrics = summarize_decay(ic, cfg["splits"]["train_end"], cfg["splits"]["valid_end"])
            metrics.update(regime_summary(ic, regimes))
            weights = long_short_weights(ef.factor, cfg["long_frac"], cfg["short_frac"], cfg["rebalance_freq"])
            port = portfolio_returns(weights, next_ret, cost_bps=10)
            metrics.update(perf_summary(port))
            metrics["max_corr"] = max_factor_corr(ef.factor, existing_factors)
            metrics["complexity"] = ef.complexity
            metrics["formula"] = cand.formula
            metrics["hypothesis"] = cand.hypothesis
            metrics["decayguard_score"] = decayguard_score(metrics, cfg.get("score_weights"))
            rows.append(metrics)
            existing_factors.append(ef.factor)
            factors[cand.formula] = (ef.factor, ic, port)
        except Exception as exc:
            rows.append({"formula": cand.formula, "error": str(exc), "decayguard_score": float("nan")})

    out_dir = Path("results")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("decayguard_score", ascending=False, na_position="last")
    df.to_csv(out_dir / "eval_table.csv", index=False)
    print(df[["formula", "valid_rank_ic_mean", "test_rank_ic_mean", "rank_ic_train_test_gap", "turnover", "max_corr", "complexity", "decayguard_score"]].head(10).to_string(index=False))

    # Plot the best alpha's rolling RankIC and equity curve at several costs.
    best_formula = df.iloc[0]["formula"]
    factor, ic, _ = factors[best_formula]
    ax = ic["rank_ic"].rolling(60, min_periods=20).mean().plot(title=f"Rolling RankIC: {best_formula}")
    ax.set_ylabel("60d rolling RankIC")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "rolling_rank_ic.png", dpi=160)
    plt.close()

    plt.figure()
    weights = long_short_weights(factor, cfg["long_frac"], cfg["short_frac"], cfg["rebalance_freq"])
    for bps in cfg["cost_bps"]:
        port = portfolio_returns(weights, next_ret, cost_bps=bps)
        (1 + port["net_ret"].fillna(0)).cumprod().plot(label=f"{bps} bps")
    plt.title(f"Cost Sensitivity Equity Curve: {best_formula}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "cost_sensitivity.png", dpi=160)
    plt.close()

    report = out_dir / "report.md"
    report.write_text(
        "# DecayGuard Demo Report\n\n"
        f"Best formula: `{best_formula}`\n\n"
        "## Top candidates\n\n"
        + df.head(10).to_markdown(index=False)
        + "\n\nFigures saved in `results/figures/`.\n"
    )
    print(f"\nSaved results/eval_table.csv and {report}")


if __name__ == "__main__":
    main()
