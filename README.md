# DecayGuard: LLM-assisted Alpha Mining with Out-of-Sample Decay Control

DecayGuard is a minimal research prototype for formulaic alpha mining with LLM-generated candidates and robust out-of-sample validation.

This starter repo focuses on the core quant research loop:

1. Generate formulaic alpha candidates with an LLM or a built-in fallback list.
2. Parse formulas through a safe Alpha DSL instead of arbitrary Python execution.
3. Reject invalid / overly complex / potentially leaky formulas.
4. Evaluate each alpha with IC, Rank IC, train-valid-test decay, rolling Rank IC slope, turnover, factor redundancy, and regime stability.
5. Build a simple cost-aware long-short portfolio.
6. Produce tables and figures for a research-style report.

> This project is for research and educational purposes only. It is not investment advice.

## References / inspiration

- AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay, KDD 2025.
- AlphaForge: formulaic alpha factor mining and dynamic factor combination, AAAI 2025.
- Microsoft Qlib: AI-oriented quant research platform.
- RD-Agent(Q): multi-agent quant R&D automation.

## Quickstart

```bash
conda create -n decayguard python=3.10 -y
conda activate decayguard
pip install -e .

python scripts/run_demo.py --config configs/demo.yaml
```

The demo uses synthetic OHLCV data, so it runs without downloading market data. Replace the data loader with Qlib or your own CSV after the pipeline works.

## Optional: local LLM through Ollama

```bash
ollama pull qwen3:8b
ollama serve
python scripts/generate_alphas.py --config configs/demo.yaml --out results/candidates.jsonl
```

If Ollama is not running, the script falls back to hand-written sample alpha formulas.

## Repository layout

```text
decayguard-alpha-mining/
├── configs/
├── scripts/
├── src/decayguard/
│   ├── dsl/          # parser, operators, leakage guard
│   ├── factors/      # data and factor calculation
│   ├── evaluation/   # IC, decay, regime, redundancy, scoring
│   ├── backtest/     # portfolio construction and costs
│   └── llm/          # Ollama client and prompts
└── results/
```

## Next steps

- Replace synthetic data with Qlib China-stock / US-stock data.
- Add industry neutralization.
- Add AST similarity against a persistent alpha library.
- Add LLM critic feedback loop.
- Add ablations: IC-only selection vs AST-only vs DecayGuard.
