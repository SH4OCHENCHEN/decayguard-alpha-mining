from __future__ import annotations

import numpy as np
import pandas as pd

from decayguard.dsl.parser import Node, to_tokens


def ast_jaccard(a: Node, b: Node) -> float:
    sa, sb = set(to_tokens(a)), set(to_tokens(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def max_factor_corr(new_factor: pd.DataFrame, existing: list[pd.DataFrame]) -> float:
    if not existing:
        return 0.0
    x = new_factor.stack().rename("x")
    best = 0.0
    for f in existing:
        y = f.stack().rename("y")
        joined = pd.concat([x, y], axis=1).dropna()
        if len(joined) < 100:
            continue
        corr = abs(joined["x"].corr(joined["y"]))
        if np.isfinite(corr):
            best = max(best, float(corr))
    return best
