"""Generation loops: autoregressive baseline and the speculative loop.

The speculative loop drafts gamma tokens, verifies them in ONE batched target
forward, applies the exact rejection rule per token, resamples from the
residual on rejection, and takes the free bonus token when all are accepted.
KV caches of both models are trimmed back to the accepted prefix each round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .rejection import softmax_np, verify_proposals


@dataclass
class SpecStats:
    """Bookkeeping for one speculative generation."""

    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    bonus: int = 0
    target_forwards: int = 0
    accepted_per_round: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens accepted by the target."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Mean committed tokens per verify round (accepted + 1)."""
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


def autoregressive_generate(
    model,
    prompt: Sequence[int],
    n_new: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> list[int]:
    """Baseline: one target forward per generated token (KV-cached)."""
    toks = list(prompt)
    cache = model.new_cache()
    logits = model.forward_cached(toks, cache)
    for _ in range(n_new):
        p = softmax_np(logits[-1], temperature)
        x = int(rng.choice(len(p), p=p / p.sum()))
        toks.append(x)
        if len(toks) - len(prompt) < n_new:
            logits = model.forward_cached([x], cache)
    return toks


def speculative_generate(
    target,
    drafter,
    prompt: Sequence[int],
    n_new: int,
    gamma: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    accept_all: bool = False,
    round_hook=None,
) -> tuple[list[int], SpecStats]:
    """Speculative decoding of exactly `n_new` tokens after `prompt`.

    Each round: the drafter proposes `gamma` tokens with their distributions
    q_1..q_gamma; the target runs ONE forward over the not-yet-verified suffix
    plus the proposals, yielding p_1..p_{gamma+1}; `verify_proposals` applies
    the rejection rule (or blindly accepts when `accept_all`, the deliberately
    broken variant).  Both KV caches are trimmed to the accepted prefix.

    `round_hook`, if given, is called after every round with the live
    bookkeeping state (used by the tests to audit the KV cache).

    Returns (tokens, stats); tokens = prompt + generated (truncated to n_new).
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    toks = list(prompt)
    if not toks:
        raise ValueError("prompt must be non-empty")
    stats = SpecStats()
    tcache = target.new_cache()
    t_processed = 0
    drafter.reset()
    want = len(prompt) + n_new
    while len(toks) < want:
        c = len(toks)
        proposals, q_rows = drafter.propose(toks, gamma, rng)
        feed = toks[t_processed:] + proposals
        logits = target.forward_cached(feed, tcache)
        p_rows = softmax_np(logits[-(gamma + 1) :], temperature)
        n_acc, emitted = verify_proposals(proposals, q_rows, p_rows, rng, accept_all)
        toks.extend(emitted)
        stats.rounds += 1
        stats.proposed += gamma
        stats.accepted += n_acc
        stats.bonus += 1 if n_acc == gamma else 0
        stats.target_forwards += 1
        stats.accepted_per_round.append(n_acc)
        # cache holds c + gamma entries; only c + n_acc of them are committed
        valid = c + n_acc
        tcache.trim(min(tcache.seq_len, valid))
        t_processed = tcache.seq_len
        drafter.commit(valid)
        if round_hook is not None:
            round_hook(toks=toks, tcache=tcache, drafter=drafter, n_acc=n_acc, gamma=gamma)
    return toks[:want], stats
