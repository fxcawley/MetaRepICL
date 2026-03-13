# Review Issues Tracker

Parsed from an OpenReview-style external review (March 2025). Issues are categorized by severity and status.

**Review verdict**: Reject with encouragement to resubmit (4/10).
**Core diagnosis**: The project claims "constructive proofs and empirical validation" but delivers proof sketches and numerical self-consistency checks. No trained neural networks appear anywhere. The silent failure analysis is the strongest standalone contribution.

---

## Critical Issues

### W1: No experiments with actual transformers
**Status**: OPEN (future work)
**Severity**: Critical
**Description**: Not a single experiment involves a neural network trained via SGD. Every "experiment" is analytical/numerical:
- `route_a_minimal.py`: "Deep Transformer (GD)" is a hand-written for-loop running gradient descent on a pre-computed kernel matrix.
- `state_probes.py`: Probes test whether a linear map is invertible (tautological -- `W_true @ z` is trivially recoverable by linear regression).
- `real_data_transformer.py`: Hand-coded `Q @ K.T / tau` + `softmax()` -- a Nadaraya-Watson kernel smoother, not a transformer.
- `width_rank.py`: Random Gaussian projection + ridge regression (standard compressed sensing).
- All other experiments are numerical verifications of formulas with no learned parameters.

**Required work**:
1. Train a transformer (even 2-12 layers) on ICL tasks (synthetic linear regression) via SGD
2. Probe internal states for CG variables (alpha_t, r_t, p_t)
3. Compare convergence signatures against GD-ICL baseline (von Oswald et al.)
4. Show convergence rates match CG rate `((sqrt(kappa)-1)/(sqrt(kappa)+1))^t` vs GD rate `((kappa-1)/(kappa+1))^t`
5. Head-drop ablation on trained models showing CG-specific degradation
6. Test silent failure predictions on trained transformers
7. (Stretch) Probe pre-trained LLMs (GPT-2, LLaMA) for CG state signatures during ICL

### W2: Proofs are sketches, not proofs
**Status**: OPEN (future work)
**Severity**: Critical
**Description**: Five formal items remain unresolved stubs in the proposal:
- **S1**: Tight constants for head count and width in the epsilon-approximate mat-vec -- TBD
- **S2**: Formal preconditioner realizable with token-wise channel scalings -- TBD
- **S3**: Multi-output r>1 vectorization details -- TBD
- **S4**: Stability of f* under feature perturbations phi -> phi_hat -- TBD
- **S5**: Precise causal-masking proof obligations -- TBD

The width-rank bound (theory.md) uses `\lesssim` with no explicit constants.
Route A's "theorem" is a proof sketch, not a complete proof.

**Required work**:
1. Resolve all five stubs with complete proofs
2. State Theorem 2 (width-rank) formally with explicit constants and conditions
3. Either complete the proofs and submit as a pure theory paper, or pair with real experiments

---

## Major Issues

### W3: Expressivity results have diminishing returns
**Status**: OPEN (strategic decision)
**Severity**: Major
**Description**: The ICL theory literature is saturated with "transformers can implement X" results (Bai et al. 2023 already cover ridge/Lasso/GLMs; von Oswald et al. 2023 cover GD; Akyurek et al. 2024 cover mesa-optimization). The bar for pure expressivity results is now very high. The community has moved toward asking "what do trained transformers actually do?"

**Recommendation**: Either (a) complete proofs for a pure theory paper making only expressivity claims, or (b) invest in the critical experiments to make a theory+empirics paper. The current hybrid positioning satisfies neither standard.

### W4: Overclaims in paper sections despite honest status page ~~[PARTIALLY FIXED]~~
**Status**: FIXED (this commit)
**Severity**: Major
**Description**: Persistent tension between candid status.md and paper-facing documents. Specific overclaims:
- `paper/sections/experiments.md`: "Probes recover these states with cosine similarity > 0.9 ... indicating the model explicitly instantiates these algorithmic variables" -- missing caveat that "model" is the hand-built construction.
- `docs/experiments/mechanistic_report.md`: Conclusion claims "strong evidence for the PCG mechanism hypothesis" -- not warranted given evidence is self-confirming.
- `experiments/results_transformer_real.md`: "'Route A' mechanism ... is validated on real-world data" -- the "transformer" has zero learned parameters.
- `route_a_minimal.py`: Plot label "Deep Transformer (GD)" is misleading for a GD for-loop.
- `docs/index.md`: "A constructed deep Transformer (green)" -- misleading framing.

**Fix applied**: All the above overclaims have been corrected with honest framing in this commit.

### W5: CG vs GD distinction is underspecified and self-contradictory ~~[FIXED]~~
**Status**: FIXED
**Severity**: Major
**Description**: The project's differentiating claim is CG/PCG (second-order), not GD (first-order). But `route_a_minimal.py` -- the flagship Route A experiment -- originally implemented gradient descent, not CG. CG requires conjugate directions with specific alpha/r/p updates. The theory docs (prop.md Corollary A, docs/theory/route_a.md, route_a_theorem.md) explicitly claim CG convergence on the exponential kernel, but the experiment contradicted this by running GD. Additionally, docs/theory/route_a.md line 7 said "Gradient Descent" while lines 21 and 31 said "CG" — an internal contradiction.

**Fix applied**:
1. `route_a_minimal.py`: Replaced GD with CG as the primary iterative solver. CG now converges to the oracle in ~10 steps while GD barely reduces error in the same number of steps — directly validating the CG rate advantage. GD is retained as a comparison baseline. Updated parameters (tau=10.0, lam=1.0) to produce moderate condition number (kappa~27) where the CG/GD gap is visible. Plot now shows both CG and GD predictions (left) and convergence trajectories (right).
2. `docs/theory/route_a.md`: Fixed line 7 from "Gradient Descent" to "Conjugate Gradient", resolving the internal contradiction.
3. `paper/sections/theory.md`: Fixed proof sketch step 3 from vague "gradient-based updates" to explicit "Conjugate Gradient updates (as in Lemma 3, with K -> K_tilde)".
4. `paper/sections/experiments.md`: Updated Route A results section to describe CG convergence comparison.
5. `docs/index.md`: Updated Route A key results description for CG.
6. Config files (`configs/route_a.yaml`, `configs/model/route_a_head.yaml`) and `run_eval.py` updated with new default parameters.

---

## Minor Issues

### W6: Route B centering requirement
**Status**: OPEN (future work)
**Severity**: Minor
**Description**: The softmax-to-linear approximation (`route_b_approx.py`) explicitly centers features and targets. In a real transformer, nothing enforces this centering. The Route B bound document acknowledges this, but no formal connection between LayerNorm and the required centering is established.

**Required work**: Establish a formal connection between LayerNorm and the centering assumption, or acknowledge this as a limitation in the Route B bound.

### M1: ill_conditioned.py preconditioned CG comparison ~~[FIXED]~~
**Status**: FIXED (this commit)
**Severity**: Minor
**Description**: The preconditioned CG experiment solves a different problem than standard CG (the effective regularizer changes to `lam * (diag(K) + lam)`). This makes the convergence comparison somewhat apples-to-oranges.

**Fix applied**: Added explicit comment documenting the apples-to-oranges nature and caveating the interpretation.

### M2: heads_temp.py shared vs. dedicated noise test ~~[FIXED]~~
**Status**: FIXED (this commit)
**Severity**: Minor
**Description**: The 1e-6-scale noise used to simulate independent parameters is negligible by construction, making the test uninformative.

**Fix applied**: Increased noise scale to 1e-2 (meaningful perturbation) and added explicit docstring caveat that this tests sensitivity to feature perturbation, not true independent learned parameters.

### M3: No LaTeX paper build
**Status**: OPEN (future work)
**Severity**: Minor
**Description**: All "paper sections" are markdown. No submission-ready PDF pipeline exists.

### M4: Real data results are weak
**Status**: OPEN (acknowledged)
**Severity**: Minor
**Description**: Best R^2 = 0.21 on AG News with Ridge on BoW, with multiple datasets showing negative R^2. This is consistent with the kernel smoother being a weak baseline, not evidence for or against the theory.

### M5: Compute budgets reveal no model training
**Status**: ACKNOWLEDGED
**Severity**: Minor
**Description**: All experiments run in <5 minutes on CPU with <4GB VRAM, consistent with numerical simulations rather than model training/evaluation. This is a direct consequence of W1.

### M6: state_probes.py probe circularity ~~[FIXED]~~
**Status**: FIXED (this commit)
**Severity**: Minor (already acknowledged in mechanistic_report.md, but code lacked explicit warnings)
**Description**: The probe applies `W_true @ z` (a known random linear map) to CG states, then fits a linear probe to recover `z`. This is inverting a random matrix -- a tautology from linear algebra, not a finding about transformers.

**Fix applied**: Added prominent docstring and inline warnings about circularity. Updated plot title.

---

## Review Strengths Acknowledged

- **S1**: Honest self-assessment in status.md (positioning table, risk table)
- **S2**: Silent failure analysis is the strongest contribution (genuine numerical analysis with actionable findings)
- **S3**: Width-rank connection is a reasonable theoretical direction
- **S4**: Clean code architecture and infrastructure

---

## Recommended Resubmission Strategy

1. **Option A (Pure theory)**: Complete all five proof stubs. Remove "empirical validation" language. Frame as expressivity results with constructive proofs only. Silent failure analysis can be a standalone contribution.

2. **Option B (Theory + empirics)**: Train even small transformers (2-12 layers, synthetic linear regression) and probe them. This is the critical experiment the project identifies but hasn't done. Compare CG vs GD convergence rates in trained models.

3. **Silent failure as standalone**: The silent failure analysis could be a workshop paper on numerical analysis of softmax attention as a kernel method, framed as analysis of a mathematical construction.
