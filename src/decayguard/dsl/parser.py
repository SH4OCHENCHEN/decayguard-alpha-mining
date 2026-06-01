from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

ALLOWED_VARIABLES = {"Open", "High", "Low", "Close", "Volume", "VWAP", "Return_1d"}
ALLOWED_OPERATORS = {
    "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Log", "Min", "Max",
    "Rank", "ZScore", "TsMean", "TsStd", "TsRank", "Delta", "Corr", "DecayLinear",
}
WINDOW_OPERATORS = {"TsMean", "TsStd", "TsRank", "Delta", "DecayLinear"}
CORR_OPERATORS = {"Corr"}


@dataclass(frozen=True)
class Node:
    op: str | None = None
    args: tuple[Any, ...] = ()
    value: str | int | float | None = None

    def is_leaf(self) -> bool:
        return self.op is None


class DSLValidationError(ValueError):
    pass


def parse_formula(formula: str) -> Node:
    try:
        expr = ast.parse(formula, mode="eval").body
    except SyntaxError as exc:
        raise DSLValidationError(f"Invalid syntax: {exc}") from exc
    return _convert(expr)


def _convert(expr: ast.AST) -> Node:
    if isinstance(expr, ast.Name):
        if expr.id not in ALLOWED_VARIABLES:
            raise DSLValidationError(f"Unknown variable: {expr.id}")
        return Node(value=expr.id)
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, (int, float)):
            return Node(value=expr.value)
        raise DSLValidationError("Only numeric constants are allowed")
    if isinstance(expr, ast.Call):
        if not isinstance(expr.func, ast.Name):
            raise DSLValidationError("Only simple operator calls are allowed")
        op = expr.func.id
        if op not in ALLOWED_OPERATORS:
            raise DSLValidationError(f"Unknown operator: {op}")
        if expr.keywords:
            raise DSLValidationError("Keyword arguments are not allowed")
        args = tuple(_convert(a) for a in expr.args)
        _validate_arity(op, args)
        return Node(op=op, args=args)
    raise DSLValidationError(f"Unsupported expression: {ast.dump(expr)}")


def _validate_arity(op: str, args: tuple[Node, ...]) -> None:
    unary = {"Neg", "Abs", "Log", "Rank", "ZScore"}
    binary = {"Add", "Sub", "Mul", "Div", "Min", "Max"}
    if op in unary and len(args) != 1:
        raise DSLValidationError(f"{op} expects 1 arg")
    if op in binary and len(args) != 2:
        raise DSLValidationError(f"{op} expects 2 args")
    if op in WINDOW_OPERATORS:
        if len(args) != 2 or not isinstance(args[1].value, int):
            raise DSLValidationError(f"{op} expects (x, integer_window)")
        window = args[1].value
        if window < 2 or window > 252:
            raise DSLValidationError(f"Invalid window for {op}: {window}")
    if op in CORR_OPERATORS:
        if len(args) != 3 or not isinstance(args[2].value, int):
            raise DSLValidationError("Corr expects (x, y, integer_window)")
        window = args[2].value
        if window < 2 or window > 252:
            raise DSLValidationError(f"Invalid window for Corr: {window}")


def ast_stats(node: Node) -> dict[str, int]:
    def rec(n: Node, depth: int) -> tuple[int, int, int]:
        if n.is_leaf():
            return 1, depth, 0
        nodes = 1
        max_depth = depth
        rolling = 1 if n.op in WINDOW_OPERATORS or n.op in CORR_OPERATORS else 0
        for a in n.args:
            c_nodes, c_depth, c_roll = rec(a, depth + 1)
            nodes += c_nodes
            max_depth = max(max_depth, c_depth)
            rolling += c_roll
        return nodes, max_depth, rolling
    nodes, depth, rolling = rec(node, 1)
    return {"nodes": nodes, "depth": depth, "rolling_ops": rolling}


def complexity(node: Node) -> float:
    s = ast_stats(node)
    return s["nodes"] + 0.5 * s["depth"] + 0.2 * s["rolling_ops"]


def to_tokens(node: Node) -> list[str]:
    if node.is_leaf():
        return [str(node.value)]
    toks = [str(node.op)]
    for a in node.args:
        toks.extend(to_tokens(a))
    return toks
