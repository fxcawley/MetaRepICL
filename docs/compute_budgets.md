## Compute Budget Table

Estimated resource requirements for each experiment in WP1–WP3.
All estimates assume a single NVIDIA A100 80GB GPU. CPU-only fallbacks are noted where applicable.

### WP0 — Infrastructure

| Experiment | VRAM | Wall-time | FLOPs | CPU Fallback |
|-----------|------|-----------|-------|-------------|
| Smoke test | <1 GB | <30s | Negligible | Yes (default) |
| Unit tests | <1 GB | <2 min | Negligible | Yes (default) |

### WP1 — Expressivity (LAT→CG, Route A/B, Width–Rank)

| Experiment | VRAM | Wall-time | FLOPs | CPU Fallback |
|-----------|------|-----------|-------|-------------|
| LAT mat-vec unit tests | <1 GB | <10s | ~10^6 | Yes |
| LAT→CG t-step stack (t≤20) | <1 GB | <30s | ~10^7 | Yes |
| Route A minimal (n=48, p=16) | <2 GB | <1 min | ~10^8 | Yes |
| Route A end-to-end | <2 GB | <1 min | ~10^8 | Yes |
| Width–rank sweep (m=4..64) | <4 GB | <5 min | ~10^9 | Yes (slower) |
| Route B approx (2-head) | <2 GB | <2 min | ~10^8 | Yes |
| Preconditioner ablation | <2 GB | <2 min | ~10^8 | Yes |

### WP2 — Mechanistic Probes and Ablations

| Experiment | VRAM | Wall-time | FLOPs | CPU Fallback |
|-----------|------|-----------|-------|-------------|
| Linear probes (100 tasks, 6 layers) | <4 GB | <5 min | ~10^9 | Yes |
| Head-drop ablation | <2 GB | <2 min | ~10^8 | Yes |
| Ill-conditioned failure mode | <2 GB | <2 min | ~10^8 | Yes |
| Knob sweeps (λ, κ, t, m grid) | <8 GB | <30 min | ~10^10 | Yes (reduced grid) |
| Temperature/normalization ablation | <4 GB | <10 min | ~10^9 | Yes |

### WP3 — Empirical Benchmarks

| Experiment | VRAM | Wall-time | FLOPs | CPU Fallback |
|-----------|------|-----------|-------|-------------|
| Baselines (ridge + GD-ICL, 3 seeds) | <1 GB | <1 min | ~10^7 | Yes |
| GLM tasks (logistic/Poisson) | <2 GB | <5 min | ~10^8 | Yes |
| Language numeric-label tasks | <8 GB | <20 min | ~10^10 | Reduced config |
| Full eval harness (all targets, 5 seeds) | <8 GB | <1 hr | ~10^11 | Reduced config |

### Scaled-Down Configurations

For environments with limited GPU (e.g., single consumer GPU with 8–16 GB VRAM) or CPU-only:

| Parameter | Full Config | Reduced Config |
|----------|------------|---------------|
| n_support | 64 | 16–32 |
| p (features) | 32 | 8–16 |
| CG steps (t) | 20 | 5–10 |
| Width sweep points | 16 | 4–8 |
| Seeds | 5 | 3 |
| Knob sweep grid | 4×4×4×4 | 2×2×2×2 |

### Notes

- All synthetic experiments (WP1, WP2) run comfortably on CPU for development and CI.
- GPU is primarily needed for the language tasks (WP3) and full sweep grids.
- Checkpoint every 500–1000 steps for experiments exceeding 10 minutes.
- Preemption-safe: all long-running experiments should use the checkpoint utilities in `src/checkpoint.py`.
