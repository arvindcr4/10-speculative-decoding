"""Speculative decoding from scratch: draft/target char-level transformers,
exact rejection-sampling verification, and an n-gram self-drafting variant."""

from .corpus import build_corpus, build_alt_corpus, Vocab
from .model import TinyTransformer, KVCache
from .toy import TableModel
from .rejection import (
    acceptance_probs,
    residual_dist,
    verify_proposals,
    speculative_marginal_samples,
    softmax_np,
)
from .drafters import ModelDrafter, NgramDrafter
from .decoding import autoregressive_generate, speculative_generate, SpecStats
from .train import train_model

__all__ = [
    "build_corpus",
    "build_alt_corpus",
    "Vocab",
    "TinyTransformer",
    "KVCache",
    "TableModel",
    "acceptance_probs",
    "residual_dist",
    "verify_proposals",
    "speculative_marginal_samples",
    "softmax_np",
    "ModelDrafter",
    "NgramDrafter",
    "autoregressive_generate",
    "speculative_generate",
    "SpecStats",
    "train_model",
]
