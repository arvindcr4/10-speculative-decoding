"""Properties of the self-drafting n-gram lookahead proposer."""

import numpy as np
import pytest

from speculative_decoding import NgramDrafter, TableModel, speculative_generate


def test_ngram_proposals_only_continuations_seen_in_context():
    """Every proposed token must be a character already present in context."""
    # context 'abcab' with a=0, b=1, c=2 over a vocab of 5
    ctx = [0, 1, 2, 0, 1]
    drafter = NgramDrafter(vocab_size=5, n=3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        proposals, q_rows = drafter.propose(ctx, 4, rng)
        assert set(proposals) <= set(ctx)
        # support of every q row is inside the context alphabet
        for row in q_rows:
            assert row.sum() == pytest.approx(1.0)
            assert set(np.nonzero(row)[0]) <= set(ctx)


def test_ngram_deterministic_continuation():
    """'abcab' has a unique bigram continuation chain: c, a, b."""
    ctx = [0, 1, 2, 0, 1]
    drafter = NgramDrafter(vocab_size=5, n=3)
    rng = np.random.default_rng(1)
    proposals, q_rows = drafter.propose(ctx, 3, rng)
    assert proposals == [2, 0, 1]  # ab->c, bc->a, ca->b, all forced
    for i, tok in enumerate(proposals):
        assert q_rows[i][tok] == pytest.approx(1.0)


def test_ngram_backoff_to_unigram():
    """Unseen suffixes back off to the context's unigram distribution."""
    ctx = [3, 3, 3, 0]  # suffix (3, 0) unseen as a bigram key; 0 never continues
    drafter = NgramDrafter(vocab_size=5, n=3)
    rng = np.random.default_rng(2)
    proposals, q_rows = drafter.propose(ctx, 1, rng)
    expected = np.zeros(5)
    expected[3] = 0.75  # unigram counts 3:3, 0:1
    expected[0] = 0.25
    np.testing.assert_allclose(q_rows[0], expected, atol=1e-12)
    assert proposals[0] in (0, 3)


def test_ngram_speculative_still_matches_target():
    """Even with the n-gram drafter the emitted law equals the target's
    (the rejection rule does not care where q comes from)."""
    P = np.array(
        [
            [0.55, 0.35, 0.10],
            [0.15, 0.25, 0.60],
            [0.30, 0.50, 0.20],
        ]
    )
    target = TableModel(P)
    rng = np.random.default_rng(3)
    n = 6000
    prompt = [0, 1, 2, 0]  # gives the drafter something to chew on
    first = np.empty(n, dtype=int)
    for i in range(n):
        toks, _ = speculative_generate(
            target, NgramDrafter(vocab_size=3, n=3), prompt, 1, 2, rng
        )
        first[i] = toks[len(prompt)]
    emp = np.bincount(first, minlength=3) / n
    exact = P[prompt[-1]]
    assert 0.5 * np.abs(emp - exact).sum() < 0.02
