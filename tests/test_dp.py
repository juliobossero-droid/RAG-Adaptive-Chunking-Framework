import itertools

import numpy as np
import pytest

from chunklab.strategies.semantic import partition_dp

ARGS = dict(max_tokens=60, min_sentences=2, max_sentences=5, ideal_sentences=3, lam=0.4)


def unit(n, d, seed):
    x = np.random.default_rng(seed).normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def direct_score(emb, tokens, parts, objective, undersize_cost=1000.0, **kw):
    span = max(kw["max_sentences"] - kw["min_sentences"], 1)
    total = 0.0
    for i, j in parts:
        n = j - i
        if n > 1:
            sims = [float(emb[a] @ emb[b]) for a in range(i, j) for b in range(a + 1, j)]
            coh = sum(sims) / len(sims)
        else:
            coh = 0.0
        q = coh - kw["lam"] * abs(n - kw["ideal_sentences"]) / span
        total += q * n if objective == "sentence_weighted" else q
        if n < kw["min_sentences"]:
            total -= undersize_cost
    return total


def compositions(n, max_part):
    if n == 0:
        yield []
        return
    for k in range(1, min(n, max_part) + 1):
        for rest in compositions(n - k, max_part):
            yield [k] + rest


@pytest.mark.parametrize("objective", ["sentence_weighted", "sum"])
@pytest.mark.parametrize("seed", range(6))
def test_dp_matches_bruteforce(objective, seed):
    n = 11
    emb = unit(n, 16, seed)
    tokens = np.random.default_rng(100 + seed).integers(8, 25, size=n)
    best = -np.inf
    for comp in compositions(n, ARGS["max_sentences"]):
        parts, i = [], 0
        for k in comp:
            parts.append((i, i + k))
            i += k
        if any(tokens[a:b].sum() > ARGS["max_tokens"] and (b - a) > 1 for a, b in parts):
            continue
        best = max(best, direct_score(emb, tokens, parts, objective, **ARGS))
    parts, value = partition_dp(emb, tokens, objective=objective, **ARGS)
    assert value == pytest.approx(best, abs=1e-9)
    assert value == pytest.approx(direct_score(emb, tokens, parts, objective, **ARGS), abs=1e-9)


def test_dp_constraints_and_partition():
    n = 40
    emb = unit(n, 32, 1)
    tokens = np.full(n, 20)
    parts, _ = partition_dp(emb, tokens, objective="sentence_weighted", **ARGS)
    assert parts[0][0] == 0 and parts[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
    assert all(tokens[i:j].sum() <= ARGS["max_tokens"] for i, j in parts)
    assert all(j - i <= ARGS["max_sentences"] for i, j in parts)


def test_dp_deterministic_and_short_docs():
    emb = unit(30, 8, 3)
    tokens = np.full(30, 10)
    a, _ = partition_dp(emb, tokens, **ARGS)
    b, _ = partition_dp(emb, tokens, **ARGS)
    assert a == b
    assert partition_dp(emb[:0], tokens[:0], **ARGS) == ([], 0.0)
    p, _ = partition_dp(emb[:1], tokens[:1], **ARGS)
    assert p == [(0, 1)]


def test_weighted_objective_does_not_prefer_more_chunks_by_construction():
    # con coherencia uniforme alta, 'sum' sobre-fragmenta respecto de 'sentence_weighted'
    n = 30
    emb = np.tile(unit(1, 8, 0), (n, 1))
    tokens = np.full(n, 5)
    kw = dict(max_tokens=1000, min_sentences=2, max_sentences=10, ideal_sentences=6, lam=0.0)
    w, _ = partition_dp(emb, tokens, objective="sentence_weighted", **kw)
    s, _ = partition_dp(emb, tokens, objective="sum", **kw)
    assert len(s) >= len(w)
