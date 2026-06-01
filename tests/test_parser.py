from decayguard.dsl.parser import parse_formula, complexity


def test_parse_valid_formula():
    node = parse_formula("Mul(Rank(Neg(Delta(Close, 5))), Rank(TsStd(Return_1d, 20)))")
    assert node.op == "Mul"
    assert complexity(node) > 0
