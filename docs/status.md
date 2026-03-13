# Project Status: Honest Assessment

## What We've Built

MetaRep provides constructive proof sketches and numerical validation that Transformer attention layers have the capacity to implement:

1. **Exponential Kernel Ridge Regression** (Route A, via softmax attention)
2. **Preconditioned Conjugate Gradient** (Route B, via linear attention)
3. **Spectral sketching** under width constraints (width-rank tradeoff)

All claims are validated on synthetic data with numerical simulations confirming internal self-consistency. No trained neural networks are involved in any experiment — all validation is on hand-built constructions. See [REVIEW_ISSUES.md](../REVIEW_ISSUES.md) for a detailed issue tracker from external review.

---

## Is This Direction Still Feasible?

**Yes, with repositioning.** The ICL-as-optimization literature has matured significantly since this project began:

### The Landscape Has Shifted

Park et al. (2024) argue convincingly that ICL is a **mixture of competing algorithms** -- retrieval, inference, unigram, bigram -- with context length and training determining which dominates. This means:

- The claim "ICL implements GD" (von Oswald et al., 2023) was always an incomplete picture
- Similarly, "ICL implements KRR" would be an overclaim
- The right framing is: **KRR/CG is one algorithm in the ICL mixture**, activated under specific conditions (regression-like tasks, sufficient context, well-conditioned data)

### What Makes MetaRep Still Valuable

1. **Second-order convergence**: Fu et al. (2023) showed empirically that Transformers achieve Newton-like convergence rates. MetaRep provides a *constructive mechanism* (CG/PCG) that explains this -- an explicit "compiler" from architecture to algorithm.

2. **Falsifiable predictions**: Unlike pure expressivity results, our probes and failure modes generate testable hypotheses. If a model is using CG, we know what to look for. If it isn't, the probes will fail -- and that's informative too.

3. **The width-rank result is novel**: No other work (to our knowledge) connects Transformer width to spectral sketching quality for ICL with formal bounds. This has practical implications for model sizing.

4. **Algorithm selection framing**: Bai et al. (2023) prove transformers can do in-context algorithm selection. MetaRep's Route A and Route B are precisely the kind of base algorithms that the selector chooses between.

### Risks and Honest Limitations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Trained models may not use CG specifically | High | Frame as expressivity + testable predictions, not as "this is what models do" |
| Synthetic-only validation | High | Real-data probing of LLMs is the critical next step |
| Park et al.'s critique of monolithic claims | Medium | We've repositioned: CG/KRR is one phase, not the whole story |
| Route B complexity vs. practical implementations | Medium | Real models likely use approximate/messy variants, not clean PCG |
| Probe circularity (construction confirms construction) | Medium | Acknowledged explicitly; probing trained models is future work |

---

## What's Done

- Constructive proof sketches: LAT to CG mapping (Lemmas A-C), Route A theorem, width-rank bound (all with unresolved stubs — see S1-S5 in [REVIEW_ISSUES.md](../REVIEW_ISSUES.md))
- Numerical validation: Route A KRR alignment, failure modes, head ablations, width sweeps (all on analytical constructions, not trained models)
- Probe demonstration: CG state recovery from constructed embeddings with train/test split and negative controls (validates construction, not trained models — see circularity caveat)
- Silent failure analysis: Genuine numerical analysis of when the softmax-KRR gap degrades (strongest standalone contribution)
- **Trained transformer experiment** (NEW): 12-layer transformer (9.5M params) trained on ICL linear regression via SGD. Probed for CG vs GD state variables. **Result: model is more GD-like than CG-like** (GD probe cos sim 0.298 vs CG 0.184). See [mechanistic report](experiments/mechanistic_report.md#5-probing-a-trained-transformer-new).
- Infrastructure: CI, reproducibility, Hydra configs, containerization

## What's Not Done

- **Probing real LLMs** (LLaMA, GPT-class) for CG state signatures
- **Mixed-kappa training**: Train on tasks with varying condition numbers where CG and GD rates diverge, for a fairer CG vs GD comparison
- **Algorithm phase detection**: When does a model switch from retrieval to inference (KRR) mode?
- **GLM extension**: Non-quadratic loss surfaces (logistic, Poisson) where CG becomes nonlinear CG
- **LaTeX paper build**: No submission-ready PDF pipeline yet
- **Complete proof stubs**: Five formal items (S1-S5) remain unresolved — tight constants, formal preconditioner, multi-output, stability, causal masking
- **Convergence rate comparison**: Show trained transformers match CG rate vs GD rate on ill-conditioned problems
- **Route B centering**: Formal connection between LayerNorm and the centering assumption required by Route B

## Will Anyone Care?

**The audience is the mechanistic interpretability + ICL theory community.** Specifically:

- Researchers studying *what algorithms Transformers implement* (the "mesa-optimizer" question)
- People building on von Oswald / Dai's GD-ICL framework who want to know *why Transformers are faster than GD*
- Anyone interested in principled model sizing for ICL tasks (the width-rank connection)
- The growing "algorithmic phases of ICL" subfield that needs precise characterizations of each phase

The key selling point is not "ICL = KRR" (which would be an overclaim). It's: **here is a precise, constructive characterization of the optimization-based phase of ICL, with falsifiable predictions and mechanistic signatures.** That's a contribution regardless of whether any specific model uses it.

---

*This project is open-source under the MIT License. We welcome contributions and critique.*
