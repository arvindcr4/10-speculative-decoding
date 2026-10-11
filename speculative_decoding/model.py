"""Tiny char-level decoder-only transformer with a preallocated KV cache.

The KV cache is what makes speculative decoding's bookkeeping interesting:
after a partial rejection the cache must be trimmed back to the accepted
prefix so generation continues from the right position.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Preallocated per-layer key/value buffers for a single sequence."""

    def __init__(self, n_layers: int, n_heads: int, max_len: int, head_dim: int):
        self.k = [torch.zeros(1, n_heads, max_len, head_dim) for _ in range(n_layers)]
        self.v = [torch.zeros(1, n_heads, max_len, head_dim) for _ in range(n_layers)]
        self.max_len = max_len
        self.length = 0  # number of valid (committed) positions

    @property
    def seq_len(self) -> int:
        """Number of tokens currently represented in the cache."""
        return self.length

    def trim(self, new_len: int) -> None:
        """Roll the cache back to `new_len` tokens (after a rejection)."""
        if not 0 <= new_len <= self.length:
            raise ValueError(f"cannot trim cache of length {self.length} to {new_len}")
        self.length = new_len


class Block(nn.Module):
    """Pre-LN transformer block with causal self-attention and a 4x MLP."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        k_buf: Optional[torch.Tensor] = None,
        v_buf: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        h, hd = self.n_heads, self.head_dim
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, h, hd).transpose(1, 2)
        k = k.view(B, T, h, hd).transpose(1, 2)
        v = v.view(B, T, h, hd).transpose(1, 2)
        if k_buf is not None:
            # incremental path: write new keys/values at [offset, offset+T)
            k_buf[:, :, offset : offset + T] = k
            v_buf[:, :, offset : offset + T] = v
            k_all = k_buf[:, :, : offset + T]
            v_all = v_buf[:, :, : offset + T]
            # query i (absolute pos offset+i) may attend keys 0..offset+i
            key_pos = torch.arange(offset + T)
            qry_pos = torch.arange(offset, offset + T)
            mask = key_pos[None, :] <= qry_pos[:, None]  # (T, offset+T)
            att = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        else:
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = att.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """Char-level decoder-only transformer with learned position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def new_cache(self) -> KVCache:
        """A fresh empty KV cache sized for this model."""
        return KVCache(self.n_layers, self.n_heads, self.max_len, self.d_model // self.n_heads)

    def forward(self, idx: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        """Return logits (B, T, V). With `cache`, `idx` holds only new tokens."""
        B, T = idx.shape
        offset = cache.length if cache is not None else 0
        if offset + T > self.max_len:
            raise ValueError(f"sequence length {offset + T} exceeds max_len {self.max_len}")
        pos = torch.arange(offset, offset + T)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for i, blk in enumerate(self.blocks):
            if cache is not None:
                x = blk(x, cache.k[i], cache.v[i], offset)
            else:
                x = blk(x)
        logits = self.head(self.ln_f(x))
        if cache is not None:
            cache.length += T
        return logits

    @torch.no_grad()
    def forward_cached(self, tokens: Sequence[int], cache: KVCache) -> np.ndarray:
        """Feed new `tokens` through the cache; return logits (T, V) as numpy."""
        idx = torch.tensor([list(tokens)], dtype=torch.long)
        return self.forward(idx, cache)[0].numpy()

    def n_params(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())
