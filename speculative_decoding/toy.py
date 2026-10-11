"""Hand-buildable first-order Markov "model" sharing the transformer interface.

Used by the tests to compare speculative sampling against exactly enumerable
target distributions, and by anyone who wants a model whose next-token law
they can write down.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class _DummyCache:
    def __init__(self) -> None:
        self.length = 0

    @property
    def seq_len(self) -> int:
        return self.length

    def trim(self, new_len: int) -> None:
        if not 0 <= new_len <= self.length:
            raise ValueError("bad trim")
        self.length = new_len


class TableModel:
    """First-order Markov model: row t of `P` is the distribution after token t."""

    def __init__(self, P: np.ndarray):
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be square (V, V)")
        if not np.allclose(P.sum(axis=1), 1.0):
            raise ValueError("rows of P must sum to 1")
        self.P = P
        self.vocab_size = P.shape[0]

    def new_cache(self) -> _DummyCache:
        """A trivial cache that only tracks committed length."""
        return _DummyCache()

    def forward_cached(self, tokens: Sequence[int], cache: _DummyCache) -> np.ndarray:
        """Log-prob rows for each fed token; bumps the dummy cache length."""
        rows = self.P[np.asarray(list(tokens), dtype=np.int64)]
        cache.length += len(tokens)
        return np.log(np.maximum(rows, 1e-30))

    def next_dist(self, last_token: int) -> np.ndarray:
        """Exact next-token distribution given the last token."""
        return self.P[last_token].copy()
