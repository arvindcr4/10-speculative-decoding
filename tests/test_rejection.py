"""Unit tests for the exact rejection-sampling verification rule."""

import numpy as np
import pytest

from speculative_decoding import acceptance_probs, residual_dist
from speculative_decoding.rejection import speculative_marginal_samples, tv_distance


P3 = np.array([0.5, 0.3, 0.2])
Q3 = np.array([0.2, 0.5, 0.3])


def test_acceptance_probs_hand_computed():
    """min(1, p/q) on a 3-token toy distribution, computed by hand."""
    acc = acceptance_probs(P3, Q3)
    # token 0: p/q = 2.5 -> capped at 1; token 1: .3/.5 = .6; token 2: .2/.3 = 2/3
    np.testing.assert_allclose(acc, [1.0, 0.6, 2.0 / 3.0], rtol=1e-12)
    # overall acceptance prob when proposing from q is sum_x min(p, q) = 0.7
    overall = float(np.sum(Q3 * acc))
    assert overall == pytest.approx(0.7, abs=1e-12)


def test_residual_distribution_hand_computed():
    """(p - q)+ normalized: only token 0 has positive residual here."""
    r = residual_dist(P3, Q3)
    np.testing.assert_allclose(r, [1.0, 0.0, 0.0], atol=1e-12)
    assert r.sum() == pytest.approx(1.0)


def test_residual_preserves_target_identity():
    """min(p,q) + (1 - sum min(p,q)) * residual == p, for random dists too."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.integers(2, 12)
        p = rng.dirichlet(np.ones(v))
        q = rng.dirichlet(np.ones(v))
        beta = np.minimum(p, q).sum()  # prob a proposal is accepted
        reconstructed = np.minimum(p, q) + (1.0 - beta) * residual_dist(p, q)
        np.testing.assert_allclose(reconstructed, p, atol=1e-12)


def test_vectorized_marginal_sampler_matches_p_and_greedy_matches_q():
    """The vectorized accept/residual sampler's law is p; with accept_all, q."""
    rng = np.random.default_rng(1)
    p = np.array([0.05, 0.55, 0.10, 0.30])
    q = np.array([0.40, 0.10, 0.30, 0.20])
    n = 40000
    emp = np.bincount(speculative_marginal_samples(p, q, n, rng), minlength=4) / n
    assert tv_distance(emp, p) < 0.015
    emp_g = np.bincount(
        speculative_marginal_samples(p, q, n, rng, accept_all=True), minlength=4
    ) / n
    assert tv_distance(emp_g, q) < 0.015
    assert tv_distance(emp_g, p) > 0.3  # and that is nowhere near the target


def test_residual_zero_mass_fallback():
    """q == p leaves zero residual mass; fall back to p (still a valid dist)."""
    p = np.array([0.4, 0.4, 0.2])
    r = residual_dist(p, p)
    np.testing.assert_allclose(r, p, atol=1e-12)
    assert r.sum() == pytest.approx(1.0)
    assert tv_distance(r, p) == pytest.approx(0.0, abs=1e-12)
