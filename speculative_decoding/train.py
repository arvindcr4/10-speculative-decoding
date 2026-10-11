"""Minimal char-LM training loop with optional mid-training snapshots."""

from __future__ import annotations

import copy
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _get_batch(
    data: np.ndarray, block: int, batch: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = rng.integers(0, len(data) - block - 1, size=batch)
    x = np.stack([data[i : i + block] for i in ix])
    y = np.stack([data[i + 1 : i + block + 1] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def train_model(
    model: torch.nn.Module,
    token_ids: np.ndarray,
    steps: int = 400,
    block: int = 128,
    batch: int = 16,
    lr: float = 3e-3,
    seed: int = 0,
    snapshot_at: Optional[Iterable[int]] = None,
    log_every: int = 100,
) -> tuple[list[float], dict[int, dict]]:
    """Train `model` on a 1-D array of token ids.

    Returns (loss_history, snapshots) where snapshots maps step -> deep-copied
    state_dict taken BEFORE that step (so 0 = untrained weights).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    snap_steps = set(snapshot_at or [])
    snapshots: dict[int, dict] = {}
    losses: list[float] = []
    model.train()
    for step in range(steps):
        if step in snap_steps:
            snapshots[step] = copy.deepcopy(model.state_dict())
        x, y = _get_batch(token_ids, block, batch, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:4d}  loss {loss.item():.3f}")
    if steps in snap_steps:
        snapshots[steps] = copy.deepcopy(model.state_dict())
    model.eval()
    return losses, snapshots
