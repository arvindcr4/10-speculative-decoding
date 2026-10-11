"""Distribution-level tests of the speculative loop on hand-built table models."""

import numpy as np
import pytest
from scipy import stats as sps

from speculative_decoding import (
    ModelDrafter,
    TableModel,
    speculative_generate,
)
from speculative_decoding.rejection import tv_distance


# Hand-built 3-token Markov models: rows are next-token dists after each token.
P_TARGET = np.array(
    [
        [0.60, 0.30, 0.10],
        [0.10, 0.20, 0.70],
        [0.35, 0.45, 0.20],
    ]
)
P_DRAFT = np.array(
    [
        [0.20, 0.30, 0.50],
        [0.55, 0.25, 0.20],
        [0.10, 0.10, 0.80],
    ]
)


def _run_many(n_runs, n_new, gamma, seed, accept_all=False):
    target = TableModel(P_TARGET)
    draft = TableModel(P_DRAFT)
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(n_runs):
        toks, _ = speculative_generate(
            target, ModelDrafter(draft), [0], n_new, gamma, rng, accept_all=accept_all
        )
        outs.append(toks[1:])
    return np.array(outs)


def test_speculative_matches_target_chisquare_and_tv():
    """First-token law from 8k full speculative runs must equal the target's.

    Chi-square goodness of fit plus a TV bound, fixed seed, plus the joint
    law of the first TWO emitted tokens against exact enumeration.
    """
    n = 8000
    outs = _run_many(n, n_new=2, gamma=3, seed=0)
    first = outs[:, 0]
    counts = np.bincount(first, minlength=3)
    expected = P_TARGET[0] * n
    chi2 = sps.chisquare(counts, expected)
    assert chi2.pvalue > 1e-3, f"chi-square rejects: {chi2}"
    emp = counts / n
    assert tv_distance(emp, P_TARGET[0]) < 0.02
    # joint law of (x1, x2): exact = p(x1) * p(x2 | x1)
    joint_emp = np.zeros((3, 3))
    for a, b in outs:
        joint_emp[a, b] += 1
    joint_emp /= n
    joint_exact = P_TARGET[0][:, None] * P_TARGET
    assert tv_distance(joint_emp.ravel(), joint_exact.ravel()) < 0.03


def test_all_accept_fast_path_yields_gamma_plus_one():
    """Draft == target => every proposal accepted => one round emits g+1 tokens."""
    gamma = 3
    target = TableModel(P_TARGET)
    draft = TableModel(P_TARGET)  # identical model: p == q everywhere
    rng = np.random.default_rng(1)
    toks, s = speculative_generate(target, ModelDrafter(draft), [0], gamma + 1, gamma, rng)
    assert len(toks) == 1 + gamma + 1
    assert s.rounds == 1
    assert s.accepted == gamma
    assert s.bonus == 1
    assert s.acceptance_rate == pytest.approx(1.0)


def test_exact_token_count_with_overshoot():
    """Output must contain exactly n_new tokens even when rounds overshoot."""
    for n_new, gamma, seed in [(7, 3, 2), (1, 4, 3), (10, 4, 4)]:
        outs = _run_many(1, n_new=n_new, gamma=gamma, seed=seed)
        assert outs.shape[1] == n_new


def test_acceptance_rate_matches_sum_min_pq():
    """With gamma=1 from a fixed prompt, acceptance ~= sum_x min(p, q)."""
    target = TableModel(P_TARGET)
    rng = np.random.default_rng(5)
    n = 4000
    accepted = 0
    for _ in range(n):
        _, s = speculative_generate(target, ModelDrafter(TableModel(P_DRAFT)), [0], 1, 1, rng)
        accepted += s.accepted_per_round[0]
    expected = np.minimum(P_TARGET[0], P_DRAFT[0]).sum()  # analytic acceptance
    assert accepted / n == pytest.approx(expected, abs=0.03)


def test_greedy_accept_everything_drifts_to_draft():
    """Skipping the rejection rule makes token 1 follow q, not p."""
    n = 4000
    outs = _run_many(n, n_new=1, gamma=2, seed=6, accept_all=True)
    emp = np.bincount(outs[:, 0], minlength=3) / n
    p, q = P_TARGET[0], P_DRAFT[0]
    assert tv_distance(p, q) > 0.3  # the models really disagree here
    assert tv_distance(emp, q) < 0.05  # emitted law == draft law
    assert tv_distance(emp, p) > 0.2  # visibly wrong vs the target


def test_determinism_under_seed_table():
    """Same seed => identical token stream; different seed => different."""
    a = _run_many(1, n_new=30, gamma=3, seed=7)
    b = _run_many(1, n_new=30, gamma=3, seed=7)
    c = _run_many(1, n_new=30, gamma=3, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
