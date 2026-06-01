# DecayGuard: LLM-assisted Alpha Mining with Out-of-Sample Decay Control

DecayGuard is a research prototype for mining formulaic alpha factors with LLM agents and evaluating them with robust out-of-sample diagnostics.

Unlike naive LLM alpha generation, DecayGuard ranks candidate alphas by:
- out-of-sample Rank IC
- train-valid-test IC decay
- rolling IC decay slope
- regime-conditioned stability
- turnover and transaction-cost sensitivity
- factor redundancy / crowding
- AST complexity and leakage checks
