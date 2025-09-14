## MetaRep Project Plan — Tickets, Epics, and Milestones

### Decision log (Sep 14, 2025)
- **Venue**: Anchor on AISTATS 2026 (primary). Post arXiv late Sep. ICLR 2026 considered out-of-reach unless a draft emerges in M1–M2.
- **Route B priority**: P1 (post-submission) following a solid Route-A submission draft.
- **Compute**: Assume 1× A100 80GB with 2 concurrent jobs available; preemption acceptable with frequent checkpointing. If reduced, scale configs accordingly (see capacity notes per ticket).
- **Tracking**: Use Weights & Biases. Fallback to MLflow if required by institution.
- **Licenses**: MIT (code), CC BY 4.0 (artifacts).
- **Double-blind policy**: AISTATS is double-blind. arXiv preprint allowed but must not be cited/linked in submission; defer public, author-attributed repos until after decisions or anonymize thoroughly.
- **Seeds**: 3 seeds main results; 5 seeds critical ablations (Route A vs B, width–rank).
- **Team bandwidth (initial placeholders)**: Theory Lead (20 h/wk), Eng Lead (20 h/wk), Mech-Interp Lead (15 h/wk), optional RA (10 h/wk). Owners to be assigned on board creation.
- **Masking/aggregator**: Approved (document masks explicitly).
- **Target encoding (language tasks)**: Numeric tokens for labels.

---

## Milestones and external dates

- **M1 (Sep 18, 2025)**: LAT→CG constructive demo + figure; theorem statement draft.
- **M2 (Sep 22, 2025)**: Softmax Route-A theorem draft + minimal experiment; finalize venue choice (AISTATS primary). Post arXiv preprint.
- **M3 (Oct 10, 2025)**: Route-B ε-approx operator-norm bound; first width–rank experiment.
- **M4 (Oct 31, 2025)**: Mechanistic probes & ablations complete; probe quality ≥0.9 cosine for CG states.
- **M5 (Nov 18, 2025)**: Full results, figures, and near camera-ready draft.
- **M6 (Dec 12, 2025)**: Public repo, artifact polish, slides, minimal Colab.

External dates (targets): AISTATS abstract Sep 25; paper Oct 2. ICLR dates (Sep 19/24) are stretch; proceed only if M1–M2 exceed expectations.

---

## Epic WP0 — Infrastructure and Repo

### Ticket: Initialize repo, licensing, and structure
- **Description**: Create public repo with MIT license; add `README.md`, `PROJECT_PLAN.md`, `REPRODUCIBILITY.md`, `.gitignore`, and basic directory layout (`src/`, `experiments/`, `configs/`, `data/`, `scripts/`, `notebooks/`).
- **Acceptance Criteria**: Repo public; CI green on initial commit; docs render; license headers present.
- **Deliverables**: Repo skeleton, MIT license, README outline.
- **Dependencies**: None.
- **Estimate**: 0.5 day.
- **Owner**: Eng Lead.
- **Labels**: infra, P0, M1.

### Ticket: Experiment tracking and config stack
- **Description**: Integrate Hydra/OmegaConf for hierarchical configs; set up W&B logging with offline mode toggle; define global defaults (seeds, dtype, device). Provide a `conf/` tree.
- **Acceptance Criteria**: One dry-run experiment produces a W&B run with full config tree snapshot; offline/online both work.
- **Deliverables**: `conf/` with base, dataset, model, training groups; `logging.py` utility.
- **Dependencies**: Repo initialized.
- **Estimate**: 1 day.
- **Owner**: Eng Lead.
- **Labels**: infra, P0, M1.

### Ticket: Containerization and lockfiles
- **Description**: Provide Dockerfile with CUDA base, pinned drivers, and reproducible lockfile (conda-lock or uv). Ensure parity with local env.
- **Acceptance Criteria**: `docker build` succeeds; container runs smoke test; image captures exact versions.
- **Deliverables**: `Dockerfile`, `conda-lock.yml` (or `uv.lock`), usage docs.
- **Dependencies**: Repo initialized.
- **Estimate**: 1 day.
- **Owner**: Eng Lead.
- **Labels**: infra, reproducibility, P0, M1.

### Ticket: Reproducibility hooks and CI
- **Description**: Add deterministic seeds, cudnn flags, env capture; CI checks for formatting, lint, and a 2-minute smoke test on CPU.
- **Acceptance Criteria**: `make smoke` passes locally and in CI; reproducibility metadata dumped per run.
- **Deliverables**: `scripts/smoke_test.sh`, GitHub Actions workflow, seed utilities.
- **Dependencies**: Tracking stack.
- **Estimate**: 1 day.
- **Owner**: Eng Lead.
- **Labels**: infra, reproducibility, P0, M1.

### Ticket: Precision & numerics policy and tests
- **Description**: Default runtime to float32; enforce unit tests in float64 with 1e-6 tolerances on toys; add parity tests verifying ≤1% prediction deviation float32 vs float64 on fixed seeds.
- **Acceptance Criteria**: CI job passes float64 unit tests and float32↔float64 parity tests.
- **Deliverables**: Test modules and config flags; documentation in `REPRODUCIBILITY.md`.
- **Dependencies**: CI in place.
- **Estimate**: 0.5 day.
- **Owner**: Eng Lead.
- **Labels**: infra, numerics, P0, M1.

### Ticket: GPU environment + checkpoint policy
- **Description**: Conda/env file; version pins; gradient checkpointing; checkpoint every 500–1000 steps with safe resume.
- **Acceptance Criteria**: Resume from checkpoint recovers metrics within 1% drift; environment lockfiles exist.
- **Deliverables**: `environment.yml` or `requirements.txt`, checkpoint IO utils, resume script.
- **Dependencies**: Repo and CI.
- **Estimate**: 0.5 day.
- **Owner**: Eng Lead.
- **Labels**: infra, P0, M1.

### Ticket: Compute budget table
- **Description**: For each planned experiment, document max VRAM, wall-time, and FLOPs estimate; provide small-config fallbacks when GPU slots are constrained.
- **Acceptance Criteria**: `docs/compute_budgets.md` lists budgets for WP1–WP3 experiments; referenced by tickets.
- **Deliverables**: `docs/compute_budgets.md`.
- **Dependencies**: Baseline configs.
- **Estimate**: 0.5 day.
- **Owner**: Eng Lead.
- **Labels**: infra, planning, P0, M1.

### Ticket: Seed protocol and pre-commit hook
- **Description**: Enforce seeded runs and config hashes; add pre-commit hook that blocks runs without an explicit seed and config fingerprint in logs.
- **Acceptance Criteria**: Hook active; tables report mean±95% CI with listed seeds.
- **Deliverables**: `.pre-commit-config.yaml`, seed enforcement script.
- **Dependencies**: Tracking stack.
- **Estimate**: 0.5 day.
- **Owner**: Eng Lead.
- **Labels**: infra, reproducibility, P0, M3.

---

## Epic WP1 — Expressivity (LAT→CG, Softmax Route A/B, Width–Rank)

### Ticket: LAT mat-vec construction (Lemma A)
- **Description**: Implement linear-attention head that realizes `(Kp)_j` via q/k/v over support tokens.
- **Acceptance Criteria**: Unit tests validate equality to reference mat-vec within 1e-6 on float64 toy data.
- **Deliverables**: `src/lat/matvec.py`, tests.
- **Dependencies**: Infra.
- **Estimate**: 1.5 days.
- **Owner**: Theory Lead (impl support by Eng).
- **Labels**: theory, LAT, P0, M1.

### Ticket: Aggregator reductions (Lemma B)
- **Description**: Add aggregator token that computes global reductions (∑ r_i^2, ∑ p_i (Ap)_i) and broadcasts scalars.
- **Acceptance Criteria**: Tests match CPU reference reductions; causal masking verified.
- **Deliverables**: `src/lat/aggregator.py`, mask utilities.
- **Dependencies**: LAT mat-vec.
- **Estimate**: 1 day.
- **Owner**: Theory Lead.
- **Labels**: theory, LAT, P0, M1.

### Ticket: Token-wise affine updates (Lemma C)
- **Description**: MLP + residual updates to perform CG updates for (α, r, p) per token.
- **Acceptance Criteria**: End-to-end LAT block performs a single CG step identical to CPU reference on toy problem.
- **Deliverables**: `src/lat/cg_step.py`, tests.
- **Dependencies**: Lemma A, B.
- **Estimate**: 1.5 days.
- **Owner**: Theory Lead.
- **Labels**: theory, LAT, P0, M1.

### Ticket: LAT→CG t-step stack + figure
- **Description**: Compose t steps; generate figure replicating CG convergence curve on synthetic K.
- **Acceptance Criteria**: Error vs t matches reference CG rate curve; figure added to `figures/`.
- **Deliverables**: `src/lat/cg_stack.py`, `notebooks/lat_cg_demo.ipynb`, figure.
- **Dependencies**: CG step.
- **Estimate**: 3 days.
- **Owner**: Theory Lead.
- **Labels**: theory, LAT, P0, M1.

### Ticket: Causal masking proof obligations (S5)
- **Description**: Formalize masks to prevent query→support leakage prior to final readout; include textual proof obligations.
- **Acceptance Criteria**: Written note and mask diagrams; tests verifying mask semantics.
- **Deliverables**: `docs/masking.md`, mask unit tests.
- **Dependencies**: LAT stack.
- **Estimate**: 2 days.
- **Owner**: Theory Lead.
- **Labels**: theory, documentation, P0, M2.

### Ticket: LAT→CG proof write-up
- **Description**: Formal write-up of constructive mapping for LAT→CG including mat-vec, reductions, and updates; aligns with figures and tests.
- **Acceptance Criteria**: Draft note ready for internal theory review.
- **Deliverables**: `docs/lat_to_cg_proof.md`.
- **Dependencies**: LAT stack complete.
- **Estimate**: 2 days.
- **Owner**: Theory Lead.
- **Labels**: theory, documentation, P0, M2.

### Ticket: Softmax Route A — theorem statement and mapping
- **Description**: Write theorem mapping softmax attention to exponential kernel KRR with parameters (U, τ); define conditions and residual error terms.
- **Acceptance Criteria**: Draft theorem with definitions and proof sketch; internal review sign-off.
- **Deliverables**: `docs/route_a_theorem.md`.
- **Dependencies**: Background finalized.
- **Estimate**: 3 days.
- **Owner**: Theory Lead.
- **Labels**: theory, softmax, P0, M2.

### Ticket: Softmax Route A — minimal empirical validation
- **Description**: Implement small transformer that demonstrates KRR-like smoothing consistent with exponential kernel; compare to oracle KRR; verify kernel approximation.
- **Acceptance Criteria**: (1) RMSE within 5% of oracle; (2) qualitative attention → kernel smoother alignment; (3) operator-norm bound on support set ‖K_tilde − K_exp‖₂ ≤ ε (report ε and its dependence on temperature/U).
- **Deliverables**: `experiments/route_a_minimal.py`, results artifact.
- **Dependencies**: Tracking stack, dataset generator.
- **Estimate**: 1.5 days.
- **Owner**: Eng Lead.
- **Labels**: experiment, softmax, P0, M2.

### Ticket: Width–rank theorem — statement and implications
- **Description**: Prove and document width–rank tradeoff: when width m < p, effective KRR on rank-m sketch K̂; derive query error bound via spectral tail.
- **Acceptance Criteria**: Draft theorem with clear assumptions and bound in terms of spectral tail ∑_{i>m} σ_i(K) and/or λ-dependent effective dimension; sanity-check with toy spectra.
- **Deliverables**: `docs/width_rank_theorem.md`.
- **Dependencies**: Route A theorem draft.
- **Estimate**: 3 days.
- **Owner**: Theory Lead.
- **Labels**: theory, bounds, P0, M3.

### Ticket: Width–rank empirical — low-rank sketch experiments
- **Description**: Implement experiments varying width to probe spectral tail predictions and error; compare K vs K̂.
- **Acceptance Criteria**: Plots show predicted degradation; results reproducible across 3 seeds; report both prediction error and tr(K (K+λI)^{-1}).
- **Deliverables**: `experiments/width_rank.py`, figures.
- **Dependencies**: Width–rank theorem statement.
- **Estimate**: 2 days.
- **Owner**: Eng Lead.
- **Labels**: experiment, bounds, P0, M3.

### Ticket: Diagonal/token-wise preconditioner (S2)
- **Description**: Add realizable diagonal preconditioner via token-wise channel scaling; analyze ρ(P⁻¹A) and empirical effect.
- **Acceptance Criteria**: Measured convergence improvement consistent with analysis; ablation results.
- **Deliverables**: `src/lat/preconditioner.py`, `experiments/precond.py`.
- **Dependencies**: CG stack.
- **Estimate**: 1.5 days.
- **Owner**: Theory Lead (with Eng support).
- **Labels**: theory, ablation, P1, M3.

### Ticket: Route B — ε-approx unnormalized mat-vec bound
- **Description**: Derive operator-norm bound for two-head + MLP rescaling approximating dot-product mat-vec; target per-iter ε ≤ 1e−2; quantify total O(ε t) error.
- **Acceptance Criteria**: Bound includes explicit ε ≤ 1e−2 target per mat-vec and a cumulative guarantee ‖f_t − f*‖ ≤ C(κ) · ((√κ − 1)/(√κ + 1))^t + O(ε t); internal review.
- **Deliverables**: `docs/route_b_bound.md`.
- **Dependencies**: Route A complete.
- **Estimate**: 3 days.
- **Owner**: Theory Lead.
- **Labels**: theory, softmax, P1, M3.

### Ticket: Route B — prototype implementation and test
- **Description**: Implement two-head rescaling + small MLP; validate mat-vec approximation error ≤ 1e-2 per iter on toy problems.
- **Acceptance Criteria**: Empirical operator error ≤ target; cumulative error matches O(ε t).
- **Deliverables**: `src/softmax/route_b.py`, tests, small experiment.
- **Dependencies**: Bound note.
- **Estimate**: 2 days.
- **Owner**: Eng Lead.
- **Labels**: experiment, softmax, P1, M3.

---

## Epic WP2 — Mechanistic Probes and Falsification

### Ticket: Linear probes for CG state (α, r, p)
- **Description**: Train linear readouts to recover CG state from designated layers/heads; specify probe locations.
- **Acceptance Criteria**: Cosine similarity ≥ 0.9 on validation tasks; include a negative-control probe at an unrelated head/location expected to fail, confirming specificity.
- **Deliverables**: `experiments/probes/state_probes.py`, probe weights.
- **Dependencies**: LAT stack and softmax model.
- **Estimate**: 2 days.
- **Owner**: Mech-Interp Lead.
- **Labels**: mechanistic, probes, P0, M4.

### Ticket: Ablation — remove mat-vec/aggregator head
- **Description**: Drop head(s) corresponding to mat-vec/aggregator; measure degradation consistent with CG missing operations.
- **Acceptance Criteria**: Predictable failure signatures observed; documented figures.
- **Deliverables**: `experiments/ablations/head_drop.py`, plots.
- **Dependencies**: Probes set up.
- **Estimate**: 1 day.
- **Owner**: Mech-Interp Lead.
- **Labels**: mechanistic, ablation, P0, M4.

### Ticket: Failure-mode demo — ill-conditioned K and preconditioning
- **Description**: Construct ill-conditioned kernels where CG stalls; demonstrate improvement with diagonal preconditioner consistent with analysis.
- **Acceptance Criteria**: Clear stall-and-recover curves; narrative links to κ and preconditioner theory.
- **Deliverables**: `experiments/failure_modes/ill_conditioned.py`, plots.
- **Dependencies**: Preconditioner ticket.
- **Estimate**: 1 day.
- **Owner**: Mech-Interp Lead (Theory support).
- **Labels**: mechanistic, ablation, P1, M4.

### Ticket: Knob sweeps (λ, κ via conditioning, depth t, width m)
- **Description**: Systematic sweeps; verify predicted scaling and failure when κ large; include temperature/normalization ablations.
- **Acceptance Criteria**: Summary tables and plots; CI-friendly subset defined.
- **Deliverables**: `experiments/sweeps/` configs, results tables.
- **Dependencies**: Models and probes.
- **Estimate**: 2 days.
- **Owner**: Mech-Interp Lead (Eng support).
- **Labels**: mechanistic, sweeps, P0, M4.

### Ticket: Visualization utilities and report
- **Description**: Attention maps, probe trajectories, CG-rate overlays; produce `mechanistic_report.md`.
- **Acceptance Criteria**: Clear figures; report compiles.
- **Deliverables**: `src/viz/`, `docs/mechanistic_report.md`.
- **Dependencies**: Sweeps complete.
- **Estimate**: 1 day.
- **Owner**: Mech-Interp Lead.
- **Labels**: mechanistic, documentation, P0, M4.

---

## Epic WP3 — Empirical ICL Benchmarks

### Ticket: Synthetic linear regression generator
- **Description**: Generator for controllable spectra, noise, and κ; produces support/query splits and ground-truth KRR.
- **Acceptance Criteria**: Deterministic generation under seed; KRR oracle matches closed-form.
- **Deliverables**: `src/data/synth_linear.py`.
- **Dependencies**: Infra.
- **Estimate**: 1 day.
- **Owner**: Eng Lead.
- **Labels**: data, P0, M2.

### Ticket: GLM tasks (logistic/Poisson)
- **Description**: Controlled GLM ICL tasks with known φ; supports meta-representation alignment tests.
- **Acceptance Criteria**: Baselines run; metrics defined.
- **Deliverables**: `src/data/glm.py`, configs.
- **Dependencies**: Linear generator.
- **Estimate**: 1 day.
- **Owner**: Eng Lead.
- **Labels**: data, P0, M2.

### Ticket: Language-style numeric-label tasks
- **Description**: Small corpora with numeric labels; tokenizer configuration for numeric tokens; data loader.
- **Acceptance Criteria**: End-to-end prompt works; evaluation extracts numeric predictions reliably; log token-level accuracy and numeric MAE; forbid JSON encoding for submission path (numeric tokens only).
- **Deliverables**: `src/data/lang_numeric.py`, tokenizer setup.
- **Dependencies**: Infra.
- **Estimate**: 1.5 days.
- **Owner**: Eng Lead.
- **Labels**: data, language, P1, M3.

### Ticket: Baselines — ridge oracle and GD-ICL
- **Description**: Implement closed-form ridge on frozen φ and GD-ICL baselines with fixed/learned step sizes.
- **Acceptance Criteria**: Baselines reproduce literature performance on synthetic tasks.
- **Deliverables**: `experiments/baselines/`.
- **Dependencies**: Data generators.
- **Estimate**: 1.5 days.
- **Owner**: Eng Lead.
- **Labels**: baselines, P0, M2.

### Ticket: Evaluation harness and metrics
- **Description**: RMSE vs depth t; agreement with KRR oracle; probe-recoverability metrics; latency/compute tracking.
- **Acceptance Criteria**: One command runs eval and emits tables/plots; supports 3/5 seeds.
- **Deliverables**: `src/eval/metrics.py`, `experiments/run_eval.py`.
- **Dependencies**: Baselines and models.
- **Estimate**: 1.5 days.
- **Owner**: Eng Lead.
- **Labels**: evaluation, P0, M3.

### Ticket: Ablations — head sharing vs dedicated mat-vec head; temperature/normalization
- **Description**: Compare shared vs dedicated heads; temperature sweeps; normalization effects.
- **Acceptance Criteria**: Plots and summary; documented recommendations.
- **Deliverables**: `experiments/ablations/heads_temp.py`.
- **Dependencies**: Eval harness.
- **Estimate**: 1 day.
- **Owner**: Eng Lead (Mech-Interp support).
- **Labels**: ablation, P1, M4.

---

## Epic WP4 — Paper, Artifacts, and Release

### Ticket: Drafting — Background and related work
- **Description**: Write Background/Motivation and related work sections aligned to `prop.md` and citations.
- **Acceptance Criteria**: Sections ready for internal review.
- **Deliverables**: `paper/sections/background.md`.
- **Dependencies**: None.
- **Estimate**: 1 day.
- **Owner**: Theory Lead.
- **Labels**: paper, P0, M2.

### Ticket: Drafting — Theorems (Route A, Width–Rank)
- **Description**: Formalize theorem statements and proofs; integrate figures.
- **Acceptance Criteria**: Passes internal theory review.
- **Deliverables**: `paper/sections/theory.md`.
- **Dependencies**: WP1 drafts.
- **Estimate**: 2 days.
- **Owner**: Theory Lead.
- **Labels**: paper, theory, P0, M3.

### Ticket: Anonymity hygiene audit (double-blind)
- **Description**: Pre-submission anonymization audit: strip PDF metadata; scrub repo/commit identities; ensure no author-identifying links; if code is referenced, provide anonymized repo or “will be released upon acceptance” statement.
- **Acceptance Criteria**: Checklist completed; independent reviewer signs off; no author leakage found.
- **Deliverables**: `docs/anonymity_checklist.md`, sanitized submission package.
- **Dependencies**: Draft ready.
- **Estimate**: 0.5 day.
- **Owner**: Theory Lead (with Eng support).
- **Labels**: paper, policy, P0, M2.

### Ticket: OpenReview packaging check
- **Description**: Automated script to validate page limits, fonts, margins, anonymity, and presence of forbidden links; runs pre-submission.
- **Acceptance Criteria**: Script flags zero errors on final package; included in CI as a manual workflow.
- **Deliverables**: `scripts/check_openreview_package.py`.
- **Dependencies**: Draft build system.
- **Estimate**: 0.5 day.
- **Owner**: Eng Lead.
- **Labels**: paper, tooling, P0, M2.

### Ticket: Drafting — Experiments and mechanistic section
- **Description**: Write empirical results, probes, and ablations sections; include figures/tables.
- **Acceptance Criteria**: Complete narrative with references to artifacts.
- **Deliverables**: `paper/sections/experiments.md`.
- **Dependencies**: WP2–WP3 results.
- **Estimate**: 2 days.
- **Owner**: Mech-Interp Lead (Eng support).
- **Labels**: paper, P0, M5.

### Ticket: ArXiv preprint (late Sep)
- **Description**: Prepare LaTeX or md-to-PDF; ensure reproducibility appendix; upload.
- **Acceptance Criteria**: Preprint live on arXiv; commit tag recorded; ensure AISTATS submission does not cite or link the preprint.
- **Deliverables**: `paper/` build system; arXiv submission.
- **Dependencies**: M2 drafts.
- **Estimate**: 0.5 day.
- **Owner**: Theory Lead.
- **Labels**: release, P0, M2.

### Ticket: Public repo artifacts and checkpoints
- **Description**: Release small pretrained checkpoints and instructions; verify weights load and reproduce key plots.
- **Acceptance Criteria**: External user reproduces at least one figure end-to-end; policy check completed — no public, author-attributed repo before decisions unless anonymized.
- **Deliverables**: `REPRODUCIBILITY.md` updates, `scripts/reproduce_figure_1.sh`.
- **Dependencies**: WP1–WP3 code.
- **Estimate**: 1 day.
- **Owner**: Eng Lead.
- **Labels**: release, reproducibility, P0, M6.

### Ticket: Slides and minimal Colab
- **Description**: Prepare talk slides; Colab notebook for LAT→CG demo.
- **Acceptance Criteria**: Slides reviewed; Colab runs within 10 minutes CPU/GPU free tier.
- **Deliverables**: `slides/`, `colab/lat_cg_demo.ipynb`.
- **Dependencies**: LAT demo.
- **Estimate**: 1 day.
- **Owner**: Mech-Interp Lead.
- **Labels**: comms, P1, M6.

---

## Cross-cutting risks and mitigations

- **Softmax normalization leakage**: Prefer Route A (exponential kernel) for main claims; Route B as controlled approximation with ε ≤ 1e-2/iter.
- **Representation drift**: Freeze φ for theory-consistent runs; separate runs co-train φ; log φ drift metrics.
- **Ill-conditioning**: Use diagonal preconditioner; manage λ–accuracy trade-off; document κ sensitivity.
- **Compute shortfall**: Provide scaled configs (reduced width/depth, synthetic-only) and CPU smoke paths.

---

## Labels and conventions

- **Priority**: P0 (submission-critical), P1 (post-submission or nice-to-have).
- **Milestones**: M1–M6 as above; tickets list their target milestone.
- **Owners**: Theory Lead, Eng Lead, Mech-Interp Lead, RA. Replace with names on project board.


