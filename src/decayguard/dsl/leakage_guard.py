from __future__ import annotations

from .parser import Node, DSLValidationError, ast_stats

FORBIDDEN_TOKENS = {"FutureReturn", "Label", "Target", "Y", "shift(-", "lead", "tomorrow"}


def static_leakage_check(formula: str, node: Node, max_nodes: int = 80, max_depth: int = 12) -> None:
    lowered = formula.lower()
    for tok in FORBIDDEN_TOKENS:
        if tok.lower() in lowered:
            raise DSLValidationError(f"Potential leakage token found: {tok}")
    stats = ast_stats(node)
    if stats["nodes"] > max_nodes:
        raise DSLValidationError(f"Formula too complex: nodes={stats['nodes']} > {max_nodes}")
    if stats["depth"] > max_depth:
        raise DSLValidationError(f"Formula too deep: depth={stats['depth']} > {max_depth}")
