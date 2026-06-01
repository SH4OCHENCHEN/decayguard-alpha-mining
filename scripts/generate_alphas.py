#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from decayguard.llm.ollama import call_ollama, parse_json_list
from decayguard.llm.prompts import ALPHA_GENERATION_PROMPT

FALLBACK = [
    {"hypothesis": "Short-term reversal after sharp price drops.", "formula": "Rank(Neg(Delta(Close, 5)))", "rationale": "Recent losers may mean-revert.", "expected_regime": "high_vol", "failure_mode": "strong momentum market", "expected_turnover": "medium"},
    {"hypothesis": "High recent volatility amplifies reversal.", "formula": "Mul(Rank(Neg(Delta(Close, 5))), Rank(TsStd(Return_1d, 20)))", "rationale": "Liquidity shocks create temporary dislocations.", "expected_regime": "high_vol", "failure_mode": "persistent crash", "expected_turnover": "medium"},
    {"hypothesis": "Sustained trend continuation.", "formula": "Rank(Delta(TsMean(Close, 5), 20))", "rationale": "Smoothed price trend may persist.", "expected_regime": "bull", "failure_mode": "choppy market", "expected_turnover": "low"},
    {"hypothesis": "Volume-price divergence indicates pressure.", "formula": "Neg(Corr(Return_1d, Volume, 20))", "rationale": "Rising volume on weak returns can signal distribution.", "expected_regime": "bear", "failure_mode": "news-driven volume", "expected_turnover": "medium"},
    {"hypothesis": "Low-vol stocks with positive drift outperform.", "formula": "Mul(Rank(Delta(Close, 20)), Rank(Neg(TsStd(Return_1d, 20))))", "rationale": "Quality-like trend with lower noise.", "expected_regime": "low_vol", "failure_mode": "risk-on reversal", "expected_turnover": "low"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--out", default="results/candidates.jsonl")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    n = int(cfg.get("llm", {}).get("n_candidates", 20))

    items = None
    try:
        prompt = ALPHA_GENERATION_PROMPT.format(n=n)
        text = call_ollama(
            prompt,
            model=cfg["llm"].get("model", "qwen3:8b"),
            url=cfg["llm"].get("url", "http://localhost:11434/api/generate"),
            temperature=cfg["llm"].get("temperature", 0.7),
        )
        items = parse_json_list(text)
        print(f"Generated {len(items)} alphas from Ollama.")
    except Exception as exc:
        print(f"[WARN] Ollama generation failed, using fallback candidates: {exc}")
        items = FALLBACK

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
