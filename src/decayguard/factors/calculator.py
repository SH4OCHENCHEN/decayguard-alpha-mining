from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from decayguard.dsl.parser import parse_formula, complexity, Node
from decayguard.dsl.leakage_guard import static_leakage_check
from decayguard.dsl.operators import evaluate_node, clean_factor


@dataclass
class AlphaCandidate:
    formula: str
    hypothesis: str = ""
    rationale: str = ""
    expected_regime: str = ""
    failure_mode: str = ""
    expected_turnover: str = ""


@dataclass
class EvaluatedFactor:
    candidate: AlphaCandidate
    node: Node
    factor: pd.DataFrame
    complexity: float


def calculate_factor(candidate: AlphaCandidate, data: dict[str, pd.DataFrame]) -> EvaluatedFactor:
    node = parse_formula(candidate.formula)
    static_leakage_check(candidate.formula, node)
    raw = evaluate_node(node, data)
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("Formula must evaluate to a DataFrame")
    factor = clean_factor(raw)
    return EvaluatedFactor(candidate=candidate, node=node, factor=factor, complexity=complexity(node))
