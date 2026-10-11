"""End-to-end demo: trains the target + draft models and runs every experiment.

Writes all plots into plots/:
  training_loss.png            - char-LM loss curves for the three models
  correctness_distribution.png - THE correctness plot (spec vs direct, TV ~ 0)
  acceptance_vs_gamma.png      - acceptance rate vs draft length
  throughput.png               - tokens/sec: baseline vs speculative vs n-gram
  draft_quality.png            - acceptance rate vs draft quality (break-it #2)
  breakit_greedy_accept.png    - greedy-accept-everything drifting off target
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from speculative_decoding import (
    ModelDrafter,
    NgramDrafter,
    TinyTransformer,
    Vocab,
    autoregressive_generate,
    build_alt_corpus,
    build_corpus,
    speculative_generate,
    train_model,
)
from speculative_decoding.rejection import (
    softmax_np,
    speculative_marginal_samples,
    tv_distance,
)

# ---------------------------------------------------------------- style (dataviz)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
AQUA = "#1baf7a"  # categorical slot 3

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "text.color": INK2,
    }
)


def style_ax(ax):
    """Recessive grid + minimal spines."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f"plots/{name}", dpi=150)
    plt.close(fig)
    print(f"  wrote plots/{name}")


# ---------------------------------------------------------------- 1. setup
def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    # small matmuls thrash with many threads (and this box runs busy); 2 is fastest
    torch.set_num_threads(2)

    import os

    os.makedirs("plots", exist_ok=True)

    print("== corpora ==")
    text_a = build_corpus()
    text_b = build_alt_corpus()
    vocab = Vocab(text_a, text_b)
    ids_a = np.array(vocab.encode(text_a), dtype=np.int64)
    ids_b = np.array(vocab.encode(text_b), dtype=np.int64)
    print(f"  corpus A {len(text_a)} bytes, corpus B {len(text_b)} bytes, vocab {len(vocab)}")

    V = len(vocab)
    MAXLEN = 144
    target = TinyTransformer(V, d_model=128, n_layers=4, n_heads=4, max_len=MAXLEN)
    draft = TinyTransformer(V, d_model=64, n_layers=1, n_heads=2, max_len=MAXLEN)
    wrong_draft = TinyTransformer(V, d_model=64, n_layers=1, n_heads=2, max_len=MAXLEN)
    print(f"  target params {target.n_params():,} | draft params {draft.n_params():,}")

    # ------------------------------------------------------------ 2. training
    print("== training target (4L, d128) on corpus A ==")
    loss_t, _ = train_model(target, ids_a, steps=350, seed=0)
    print("== training draft (1L, d64) on corpus A, with snapshots ==")
    loss_d, snaps = train_model(draft, ids_a, steps=250, seed=1, snapshot_at=[0, 25, 100])
    print("== training WRONG-corpus draft (1L, d64) on corpus B ==")
    loss_w, _ = train_model(wrong_draft, ids_b, steps=250, seed=2)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(loss_t, color=BLUE, lw=2, label="target (4L, d128, corpus A)")
    ax.plot(loss_d, color=ORANGE, lw=2, label="draft (1L, d64, corpus A)")
    ax.plot(loss_w, color=AQUA, lw=2, label="wrong-corpus draft (corpus B)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats/char)")
    ax.set_title("Char-LM training loss")
    style_ax(ax)
    ax.legend(frameon=False)
    savefig(fig, "training_loss.png")

    prompt = vocab.encode("The archivist w")
    with torch.no_grad():
        logits = target(torch.tensor([prompt])).numpy()[0]
    p_exact = softmax_np(logits[-1])
    dcache = draft.new_cache()
    q_draft = softmax_np(draft.forward_cached(prompt, dcache)[-1])
    q_wrong = softmax_np(
        wrong_draft.forward_cached(prompt, wrong_draft.new_cache())[-1]
    )

    # -------------------------------------------- 3. THE correctness plot
    print("== correctness: 20k speculative next-token samples vs direct ==")
    rng = np.random.default_rng(10)
    n_samp = 20_000
    spec_samples = speculative_marginal_samples(p_exact, q_draft, n_samp, rng)
    direct_samples = rng.choice(V, size=n_samp, p=p_exact / p_exact.sum())
    emp_spec = np.bincount(spec_samples, minlength=V) / n_samp
    emp_direct = np.bincount(direct_samples, minlength=V) / n_samp
    tv_sd = tv_distance(emp_spec, emp_direct)
    tv_sp = tv_distance(emp_spec, p_exact)
    # sampling-noise floor: two independent direct-sample histograms
    floor = tv_distance(
        np.bincount(rng.choice(V, n_samp, p=p_exact), minlength=V) / n_samp,
        np.bincount(rng.choice(V, n_samp, p=p_exact), minlength=V) / n_samp,
    )
    print(f"  TV(spec, direct) = {tv_sd:.4f}   TV(spec, exact p) = {tv_sp:.4f}"
          f"   noise floor ~ {floor:.4f}")

    # cross-check with the FULL loop (real model calls), smaller n
    n_loop = 2000
    loop_first = np.empty(n_loop, dtype=int)
    rng_l = np.random.default_rng(11)
    for i in range(n_loop):
        toks, _ = speculative_generate(target, ModelDrafter(draft), prompt, 1, 4, rng_l)
        loop_first[i] = toks[len(prompt)]
    tv_loop = tv_distance(np.bincount(loop_first, minlength=V) / n_loop, p_exact)
    print(f"  full-loop cross-check ({n_loop} runs): TV(loop, exact p) = {tv_loop:.4f}"
          f"  (noise floor at n={n_loop} is larger)")

    top = np.argsort(p_exact)[::-1][:15]
    labels = ["␣" if vocab.itos[i] == " " else repr(vocab.itos[i])[1:-1] for i in top]
    xs = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(xs, emp_direct[top], width=0.78, color=BLUE, label="direct target sampling")
    ax.bar(xs, emp_spec[top], width=0.42, color=ORANGE, label="speculative decoding")
    ax.set_xticks(xs, labels)
    ax.set_xlabel(f"next token after prompt '…archivist w' (top 15 of {V})")
    ax.set_ylabel("empirical probability (20k samples)")
    ax.set_title("Speculative decoding preserves the target distribution")
    ax.text(
        0.98, 0.80,
        f"TV(spec, direct) = {tv_sd:.4f}\nTV(spec, exact target) = {tv_sp:.4f}\n"
        f"sampling-noise floor ≈ {floor:.4f}",
        transform=ax.transAxes, ha="right", va="top", color=INK,
    )
    style_ax(ax)
    ax.legend(frameon=False, loc="upper right")
    savefig(fig, "correctness_distribution.png")

    # -------------------------------------------- 4. acceptance rate vs gamma
    print("== acceptance rate vs gamma ==")
    gammas = [1, 2, 3, 4, 6, 8]
    acc_model, acc_ngram = [], []
    for g in gammas:
        rates_m, rates_n = [], []
        for seed in (20, 21, 22):
            rng = np.random.default_rng(seed)
            _, s = speculative_generate(target, ModelDrafter(draft), prompt, 110, g, rng)
            rates_m.append(s.acceptance_rate)
            rng = np.random.default_rng(seed)
            _, s = speculative_generate(target, NgramDrafter(V, n=3), prompt, 110, g, rng)
            rates_n.append(s.acceptance_rate)
        acc_model.append(np.mean(rates_m))
        acc_ngram.append(np.mean(rates_n))
        print(f"  gamma={g}: model draft {acc_model[-1]:.3f} | n-gram {acc_ngram[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(gammas, acc_model, "-o", color=BLUE, lw=2, ms=6, label="model draft (1L, d64)")
    ax.plot(gammas, acc_ngram, "-o", color=ORANGE, lw=2, ms=6, label="n-gram self-draft (n=3)")
    ax.set_xlabel("draft length γ (tokens proposed per round)")
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(0, 1)
    ax.set_title("Acceptance rate vs γ")
    style_ax(ax)
    ax.legend(frameon=False)
    savefig(fig, "acceptance_vs_gamma.png")

    # -------------------------------------------- 5. throughput
    print("== throughput (tokens/sec, single CPU thread) ==")
    torch.set_num_threads(1)
    n_gen, reps = 110, 5

    def timed(fn):
        fn(np.random.default_rng(29))  # warmup (allocations, lazy init)
        times = []
        for r in range(reps):
            t = time.perf_counter()
            fn(np.random.default_rng(30 + r))
            times.append(time.perf_counter() - t)
        # min is the standard contended-machine estimator: other processes only
        # ever ADD time, so the fastest rep is closest to the true cost
        return n_gen / min(times)

    tps_base = timed(lambda rng: autoregressive_generate(target, prompt, n_gen, rng))
    print(f"  baseline autoregressive: {tps_base:6.1f} tok/s")
    tps_model, tps_ngram = [], []
    for g in gammas[1:]:  # gamma >= 2
        tm = timed(
            lambda rng: speculative_generate(target, ModelDrafter(draft), prompt, n_gen, g, rng)
        )
        tn = timed(
            lambda rng: speculative_generate(target, NgramDrafter(V, n=3), prompt, n_gen, g, rng)
        )
        tps_model.append(tm)
        tps_ngram.append(tn)
        print(f"  gamma={g}: model draft {tm:6.1f} tok/s | n-gram {tn:6.1f} tok/s")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(tps_base, color=MUTED, lw=1.5, ls="--")
    ax.text(gammas[1], tps_base * 1.03, f"autoregressive baseline ({tps_base:.0f} tok/s)",
            color=INK2, fontsize=9)
    ax.plot(gammas[1:], tps_model, "-o", color=BLUE, lw=2, ms=6, label="speculative, model draft")
    ax.plot(gammas[1:], tps_ngram, "-o", color=ORANGE, lw=2, ms=6, label="speculative, n-gram draft")
    ax.set_xlabel("draft length γ")
    ax.set_ylabel("wall-clock tokens / sec (CPU, 1 thread)")
    ax.set_title("Decoding throughput: one batched verify per γ drafts")
    style_ax(ax)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "throughput.png")

    # -------------------------------------------- 6. acceptance vs draft quality
    print("== acceptance vs draft quality (gamma=4) ==")
    quality = []
    names = ["untrained", "25 steps", "100 steps", "250 steps", "wrong corpus"]
    drafts = []
    for step in (0, 25, 100):
        m = TinyTransformer(V, d_model=64, n_layers=1, n_heads=2, max_len=MAXLEN)
        m.load_state_dict(snaps[step])
        m.eval()
        drafts.append(m)
    drafts += [draft, wrong_draft]
    for name, m in zip(names, drafts):
        rates = []
        for seed in (40, 41):
            rng = np.random.default_rng(seed)
            _, s = speculative_generate(target, ModelDrafter(m), prompt, 110, 4, rng)
            rates.append(s.acceptance_rate)
        quality.append(np.mean(rates))
        print(f"  {name:12s}: acceptance {quality[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = [BLUE] * 4 + [ORANGE]
    bars = ax.bar(names, quality, width=0.62, color=colors)
    for b, v in zip(bars, quality):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", color=INK)
    ax.set_ylabel("acceptance rate (γ=4)")
    ax.set_ylim(0, 1)
    ax.set_title("Draft quality drives acceptance; a wrong-corpus draft tanks it")
    style_ax(ax)
    savefig(fig, "draft_quality.png")

    # -------------------------------------------- 7. break-it: greedy accept
    print("== break-it: greedy-accept-everything (skip the rejection rule) ==")
    rng = np.random.default_rng(50)
    variants = {
        "speculative\n(trained draft)": (q_draft, False),
        "speculative\n(wrong-corpus draft)": (q_wrong, False),
        "greedy accept\n(trained draft)": (q_draft, True),
        "greedy accept\n(wrong-corpus draft)": (q_wrong, True),
    }
    tvs, emps = [], {}
    for name, (q, greedy) in variants.items():
        samp = speculative_marginal_samples(p_exact, q, n_samp, rng, accept_all=greedy)
        emp = np.bincount(samp, minlength=V) / n_samp
        emps[name] = emp
        tvs.append(tv_distance(emp, p_exact))
        print(f"  {name.replace(chr(10), ' '):38s} TV vs target = {tvs[-1]:.4f}")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 4.0), width_ratios=[1.25, 1])
    greedy_emp = emps["greedy accept\n(wrong-corpus draft)"]
    axl.bar(xs, p_exact[top], width=0.78, color=BLUE, label="exact target p")
    axl.bar(xs, greedy_emp[top], width=0.42, color=ORANGE,
            label="greedy accept, wrong-corpus draft")
    axl.set_xticks(xs, labels)
    axl.set_xlabel("next token (top 15 by target prob)")
    axl.set_ylabel("probability")
    axl.set_title("Silent wrongness: output drifts to the draft's law")
    style_ax(axl)
    axl.legend(frameon=False)

    colors = [BLUE, BLUE, ORANGE, ORANGE]
    names4 = ["speculative\ntrained\ndraft", "speculative\nwrong-corpus\ndraft",
              "greedy\ntrained\ndraft", "greedy\nwrong-corpus\ndraft"]
    bars = axr.bar(names4, tvs, width=0.62, color=colors)
    for b, v in zip(bars, tvs):
        axr.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", color=INK)
    axr.axhline(floor, color=MUTED, lw=1.2, ls="--")
    axr.text(0.02, 0.12, f"dashed: sampling-noise floor ≈ {floor:.3f}",
             transform=axr.transAxes, color=INK2, fontsize=8)
    axr.set_ylabel("TV distance to exact target")
    axr.set_title("Rejection rule on (blue) vs off (orange)")
    axr.tick_params(axis="x", labelsize=8)
    style_ax(axr)
    savefig(fig, "breakit_greedy_accept.png")

    # -------------------------------------------- 8. sample text
    rng = np.random.default_rng(60)
    toks, s = speculative_generate(target, ModelDrafter(draft), prompt, 110, 4, rng)
    print("== sample (speculative, gamma=4) ==")
    print("  " + repr(vocab.decode(toks)))
    print(f"  acceptance {s.acceptance_rate:.2f}, {s.tokens_per_round:.2f} tokens/round, "
          f"{s.target_forwards} target forwards for {len(toks) - len(prompt)} tokens")
    print(f"== done in {time.time() - t0:.1f}s ==")


if __name__ == "__main__":
    main()
