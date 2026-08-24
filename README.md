# Speculative Decoding

> Project 10 of 34 — [Step-By-Step LLM Engineering Projects (2026 Edition)](https://x.com/TheAhmadOsman/status/2058745340895870985) by [@TheAhmadOsman](https://x.com/TheAhmadOsman)
>
> **Part VI: Decoding Turns Probabilities Into Text**

## Goal

Accelerate sequential autoregressive decoding with a smaller draft model whose proposals a larger model verifies, preserving the target distribution. Study Medusa-style extra decoding heads and lookahead decoding variants that drop the separate draft model.

## The project loop

1. **Build it.** Implement the core idea yourself before using the library version.
2. **Plot it.** Loss curves, latency curves, memory curves, attention heatmaps, routing histograms, entropy plots, or failure galleries.
3. **Break it.** Ablate the thing you just built and watch what changes.
4. **Explain it.** A short technical note: what you expected, what happened, what surprised you, what you'd try next.

## Artifacts to ship

- [ ] **Implementation** — clean code with tests
- [ ] **Notebook** — one reproducible experiment
- [ ] **Plots** — at least three charts showing behavior
- [ ] **Failure gallery** — examples where the system breaks
- [ ] **Write-up** — what I learned and what changed my mind

---

*Status: scaffolded — build in progress.*
