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

## Contribution and privacy

- Double-blind policy (AISTATS): do not push author-identifying artifacts to public before decisions. Use anonymized branches if needed.
- See issues for current work; prefer small PRs that reference tickets (e.g., "Closes #1").
