# MetaRep — In-Context Learning as Meta-Representation Alignment

This repository tracks the MetaRep project: formalizing and validating the hypothesis that transformer ICL implements kernel ridge regression (KRR) on learned hidden representations, including constructive LAT→CG mappings and softmax Route-A/B.

See `PROJECT_PLAN.md` for epics and tickets. Issues are pre-seeded from that plan; filter by labels P0/P1 and milestones M1–M6.

## Quickstart

- Environment
  - Python 3.10+
  - Optional: Docker (CUDA base) with lockfiles
- Setup
  - `conda create -n metarep python=3.10 -y && conda activate metarep`
  - `pip install -r requirements.txt`

## Repo layout

- `src/lat/` linear-attention CG blocks (LAT→CG)
- `src/data/` synthetic and GLM data generators
- `experiments/` runnable scripts for Route A/B, width–rank, probes
- `configs/` Hydra configs
- `tests/` unit tests (float64 baselines; float32 parity)
- `docs/` proofs, masking, compute budgets, anonymity checklist
- `.github/workflows/` CI and packaging checks

## How to run (MVP targets)

- Baselines (ridge oracle vs GD-ICL; prints mean±CI over seeds):
  - `make baselines`
- Width–rank (rank-m sketch via random projection; saves a plot to `figures/width_rank.png`):
  - `make width_rank`
- Softmax Route A minimal (exp-kernel KRR vs attention-induced kernel diagnostics):
  - `make route_a`

### What these validate

- Baselines check our harness and serve as reference performance.
- Width–rank validates the spectral-tail prediction: as width m increases, prediction approaches the oracle; we also log effective dimension `d_eff(λ)` per plan.
- Route A MVP demonstrates the exponential-kernel bridge and reports operator-norm proximity `‖K̃−K_exp‖₂` on supports.

These are the first public-facing artifacts to sanity-check Aim 1–2 assumptions and guide hyperparameters for the full ICL prompts and probes.

## Contribution and privacy

- Double-blind policy (AISTATS): do not push author-identifying artifacts to public before decisions. Use anonymized branches if needed.
- See issues for current work; prefer small PRs that reference tickets (e.g., "Closes #1").
