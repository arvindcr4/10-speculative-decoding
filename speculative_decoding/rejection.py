"""The exact rejection-sampling verification rule of speculative decoding.

Given a draft distribution q and a target distribution p, a token x ~ q is
accepted with probability min(1, p(x)/q(x)); on rejection we sample from the
normalized positive residual (p - q)+.  The marginal of the emitted token is
then exactly p:  min(p, q) + (1 - sum(min(p, q))) * (p - q)+ / sum((p - q)+) = p.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1e-12


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def acceptance_probs(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Elementwise acceptance probability min(1, p/q).

    Where q == 0 the token can never be proposed; by convention we return 1
    there so the vector stays well defined.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    out = np.ones_like(p)
    mask = q > _EPS
    out[mask] = np.minimum(1.0, p[mask] / q[mask])
    return out


def residual_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Normalized positive residual (p - q)+ used to resample after a rejection.

    If the residual has (numerically) zero mass — which only happens when
    q >= p everywhere, i.e. q == p — fall back to p itself.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.clip(p - q, 0.0, None)
    s = r.sum()
    if s < _EPS:
        return p / p.sum()
    return r / s


def _sample(dist: np.ndarray, rng: np.random.Generator) -> int:
    dist = np.clip(np.asarray(dist, dtype=np.float64), 0.0, None)
    return int(rng.choice(len(dist), p=dist / dist.sum()))


def verify_proposals(
    proposals: Sequence[int],
    q_rows: np.ndarray,
    p_rows: np.ndarray,
    rng: np.random.Generator,
    accept_all: bool = False,
) -> tuple[int, list[int]]:
    """Verify draft `proposals` against the target distributions.

    Args:
        proposals: the gamma draft tokens x_1..x_gamma.
        q_rows: (gamma, V) draft distributions the proposals were sampled from.
        p_rows: (gamma+1, V) target distributions at the same positions plus
            the bonus position after the last proposal.
        rng: numpy Generator driving all randomness.
        accept_all: the BROKEN variant — skip the rejection rule entirely.

    Returns:
        (n_accepted, emitted) where emitted = accepted proposals plus exactly
        one extra token (residual sample on rejection, bonus sample otherwise).
    """
    gamma = len(proposals)
    if q_rows.shape[0] != gamma or p_rows.shape[0] != gamma + 1:
        raise ValueError("q_rows must have gamma rows and p_rows gamma+1 rows")
    if accept_all:
        bonus = _sample(p_rows[gamma], rng)
        return gamma, list(proposals) + [bonus]
    emitted: list[int] = []
    for i, x in enumerate(proposals):
        p_x = p_rows[i][x]
        q_x = max(q_rows[i][x], _EPS)
        if rng.random() < min(1.0, p_x / q_x):
            emitted.append(int(x))
        else:
            emitted.append(_sample(residual_dist(p_rows[i], q_rows[i]), rng))
            return i, emitted
    bonus = _sample(p_rows[gamma], rng)
    emitted.append(bonus)
    return gamma, emitted


def speculative_marginal_samples(
    p: np.ndarray,
    q: np.ndarray,
    n: int,
    rng: np.random.Generator,
    accept_all: bool = False,
) -> np.ndarray:
    """Vectorized samples of the FIRST emitted token of a speculative round.

    For a fixed prompt, causal masking makes the position-1 quantities
    identical across trials: the draft proposes x ~ q, the target's verify
    distribution at that position is p regardless of later draft tokens.  So
    n independent speculative first tokens can be drawn without n model calls,
    using exactly the same accept/residual rule as `verify_proposals`.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    v = len(p)
    x = rng.choice(v, size=n, p=q / q.sum())
    if accept_all:
        return x
    u = rng.random(n)
    acc = u < np.minimum(1.0, p[x] / np.maximum(q[x], _EPS))
    n_rej = int((~acc).sum())
    if n_rej:
        x[~acc] = rng.choice(v, size=n_rej, p=residual_dist(p, q))
    return x


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation distance between two distributions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())
