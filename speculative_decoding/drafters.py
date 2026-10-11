"""Draft-token proposers: a small model drafter and an n-gram self-drafter.

Both return the proposals together with the exact distributions q_i they were
sampled from — the verification rule needs those to stay unbiased.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from .rejection import softmax_np


class ModelDrafter:
    """Drafts with a separate (smaller) model, keeping its own KV cache."""

    def __init__(self, model, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature
        self.cache = model.new_cache()
        self.processed = 0  # tokens of the committed sequence in the cache

    def reset(self) -> None:
        """Forget all cached state (start a new generation)."""
        self.cache = self.model.new_cache()
        self.processed = 0

    def propose(
        self, toks: Sequence[int], gamma: int, rng: np.random.Generator
    ) -> tuple[list[int], np.ndarray]:
        """Propose `gamma` tokens after `toks`; return (proposals, q_rows)."""
        toks = list(toks)
        feed = toks[self.processed :]
        if not feed:
            raise ValueError("drafter cache is already past the context")
        logits = self.model.forward_cached(feed, self.cache)
        q = softmax_np(logits[-1], self.temperature)
        proposals: list[int] = []
        q_rows = np.empty((gamma, self.model.vocab_size))
        for i in range(gamma):
            q_rows[i] = q
            x = int(rng.choice(len(q), p=q / q.sum()))
            proposals.append(x)
            if i < gamma - 1:
                logits = self.model.forward_cached([x], self.cache)
                q = softmax_np(logits[-1], self.temperature)
        self.processed = len(toks) + gamma - 1
        return proposals, q_rows

    def commit(self, valid_len: int) -> None:
        """Trim the cache to the accepted prefix after verification."""
        keep = min(self.processed, valid_len)
        self.cache.trim(keep)
        self.processed = keep


class NgramDrafter:
    """Self-drafting in the spirit of Medusa/lookahead: no second model.

    Proposals come from an n-gram table built over the target's own committed
    context (prompt + tokens generated so far).  q_i is the empirical
    continuation distribution of the longest matching suffix, backing off to
    shorter orders and finally to the unigram distribution of the context —
    so every proposed token is a character already seen in the context.
    """

    def __init__(self, vocab_size: int, n: int = 3):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.vocab_size = vocab_size
        self.n = n

    def reset(self) -> None:
        """Stateless; nothing to reset."""

    def _tables(self, toks: Sequence[int]) -> list[dict]:
        tables = []
        for k in range(self.n - 1, 0, -1):
            table: dict = defaultdict(Counter)
            for i in range(len(toks) - k):
                table[tuple(toks[i : i + k])][toks[i + k]] += 1
            tables.append((k, table))
        return tables

    def _dist_after(self, ctx: Sequence[int], tables, unigram: Counter) -> np.ndarray:
        q = np.zeros(self.vocab_size)
        for k, table in tables:
            if len(ctx) >= k:
                counts = table.get(tuple(ctx[-k:]))
                if counts:
                    for tok, c in counts.items():
                        q[tok] = c
                    return q / q.sum()
        for tok, c in unigram.items():
            q[tok] = c
        return q / q.sum()

    def propose(
        self, toks: Sequence[int], gamma: int, rng: np.random.Generator
    ) -> tuple[list[int], np.ndarray]:
        """Propose `gamma` lookahead tokens from the context's own n-grams."""
        toks = list(toks)
        if not toks:
            raise ValueError("context must be non-empty")
        tables = self._tables(toks)
        unigram = Counter(toks)
        ctx = list(toks)
        proposals: list[int] = []
        q_rows = np.empty((gamma, self.vocab_size))
        for i in range(gamma):
            q = self._dist_after(ctx, tables, unigram)
            q_rows[i] = q
            x = int(rng.choice(self.vocab_size, p=q))
            proposals.append(x)
            ctx.append(x)
        return proposals, q_rows

    def commit(self, valid_len: int) -> None:
        """Stateless; nothing to trim."""
