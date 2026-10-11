"""KV-cache correctness: incremental == full forward, trims, and the
bookkeeping audit of speculative decoding after partial rejections."""

import numpy as np
import torch

from speculative_decoding import ModelDrafter, TinyTransformer, speculative_generate


def _model(seed=0, vocab=20, d=32, layers=2, heads=2, max_len=64):
    torch.manual_seed(seed)
    return TinyTransformer(vocab, d, layers, heads, max_len).eval()


def test_cache_incremental_matches_full_forward():
    """Chunked cached forwards must reproduce the one-shot causal forward."""
    m = _model()
    toks = list(np.random.default_rng(0).integers(0, 20, size=30))
    with torch.no_grad():
        full = m(torch.tensor([toks])).numpy()[0]
    cache = m.new_cache()
    outs = []
    for chunk in (toks[:10], toks[10:11], toks[11:16], toks[16:]):
        outs.append(m.forward_cached(chunk, cache))
    np.testing.assert_allclose(np.concatenate(outs), full, atol=1e-4)
    assert cache.seq_len == 30


def test_cache_trim_then_refeed_matches_fresh_forward():
    """After trim(12) and a different suffix, logits match a fresh forward."""
    m = _model(seed=1)
    rng = np.random.default_rng(1)
    prefix = list(rng.integers(0, 20, size=20))
    cache = m.new_cache()
    m.forward_cached(prefix, cache)
    cache.trim(12)
    suffix = list(rng.integers(0, 20, size=8))
    got = m.forward_cached(suffix, cache)
    with torch.no_grad():
        ref = m(torch.tensor([prefix[:12] + suffix])).numpy()[0][12:]
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_kv_bookkeeping_after_partial_rejection():
    """During real speculative decoding, after EVERY round the target cache
    must hold exactly the committed prefix (all tokens but the newly emitted
    one), and its K/V buffers must match a from-scratch forward — i.e.
    generation continues from the right position after a rejection."""
    target = _model(seed=2, d=32, layers=2)
    draft = _model(seed=3, d=16, layers=1, heads=1)
    prompt = [1, 2, 3, 4]
    partial_rejections = []

    def hook(toks, tcache, drafter, n_acc, gamma):
        # exactly one uncommitted token (the residual/bonus sample) each round
        assert tcache.seq_len == len(toks) - 1
        assert drafter.cache.seq_len == drafter.processed <= len(toks) - 1
        if n_acc < gamma:
            partial_rejections.append(n_acc)
            # deep audit: live buffers == buffers of a from-scratch forward
            fresh = target.new_cache()
            target.forward_cached(toks[: tcache.seq_len], fresh)
            L = tcache.seq_len
            for layer in range(target.n_layers):
                np.testing.assert_allclose(
                    tcache.k[layer][:, :, :L].numpy(),
                    fresh.k[layer][:, :, :L].numpy(),
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    tcache.v[layer][:, :, :L].numpy(),
                    fresh.v[layer][:, :, :L].numpy(),
                    atol=1e-4,
                )

    rng = np.random.default_rng(4)
    toks, s = speculative_generate(
        target, ModelDrafter(draft), prompt, 40, 4, rng, round_hook=hook
    )
    assert len(toks) == len(prompt) + 40
    # untrained mismatched models must have triggered real partial rejections
    assert partial_rejections, "test needs at least one partial rejection"
    assert 0 < s.accepted < s.proposed


def test_determinism_under_seed_transformer():
    """Same numpy seed reproduces the exact same speculative output."""
    target = _model(seed=5)
    draft = _model(seed=6, d=16, layers=1, heads=1)
    runs = []
    for seed in (11, 11, 12):
        rng = np.random.default_rng(seed)
        toks, _ = speculative_generate(target, ModelDrafter(draft), [0, 1], 25, 3, rng)
        runs.append(toks)
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]
