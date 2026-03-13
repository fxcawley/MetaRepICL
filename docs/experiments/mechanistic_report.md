# Mechanistic Report: Probing the CG State (Theoretical Construction)

## Overview

This report summarizes evidence that our *hand-built* Linear Attention Transformer (LAT) construction correctly encodes Preconditioned Conjugate Gradient (PCG) states. We validate the theoretical construction by training linear probes to recover internal optimization states ($\alpha_t, r_t, p_t$) from the simulated activation dynamics.

**Scope limitation**: All results below are on our analytical construction — no trained neural networks are involved. The probes demonstrate that CG states are linearly decodable from the constructed embedding, which is a necessary (but not sufficient) condition for the theory. Whether trained transformers exhibit similar structure is the critical open question (see [status.md](../status.md)).

## 1. Probe Recovery of CG States

### Methodology
We trained linear probes $W_{probe}$ on the simulated residual stream activations $x_i^{(l)}$ of our constructive LAT model to recover the theoretical CG states:
- **Conjugate direction** $p_t$: The search direction.
- **Residual** $r_t$: The gradient of the objective $y - K\alpha$.
- **Solution** $\alpha_t$: The accumulated weights.

*Note: This experiment validates the linear decodability of the algorithm from the theoretical construction. The "activations" are a known random linear projection $W_{\text{true}} \cdot z$ of the true CG states — so the probe is inverting a random matrix, which is a standard linear algebra operation. High cosine similarity is expected by construction. Probing trained models is future work.*

### Results
- **Cosine Similarity**: Probes consistently recover the true $p_t$ and $r_t$ with cosine similarity $> 0.9$ given the constructive embedding.
- **Trajectory**: The recovery fidelity is maintained across steps.

![Probe Cosine Similarity vs Layer](../../figures/probes/cosine_sim_layer.png){ width=600 }

## 2. Failure Modes and Ill-Conditioning

### Ill-Conditioned Kernels
When the kernel condition number $\kappa(K)$ is high ($> 100$), standard CG stalls. Our experiments show that the CG algorithm's convergence slows down on such kernels, as expected from CG theory.

### Preconditioning
Introducing a diagonal preconditioner (approximated by token-wise scaling) restores convergence rates, matching theoretical predictions for $P^{-1} \approx (diag(K) + \lambda I)^{-1}$.

![CG Stall and Recovery](../../figures/failure_modes/ill_conditioned_cg.png){ width=600 }

## 3. Ablation Studies

### Head Drop
Removing the "Aggregation Head" (Head 2 in our construction, responsible for mean subtraction) drastically increases error, confirming that the specific two-head construction (Scaled Softmax - Mean) is necessary to approximate the negative residual update correctly.

![Route B Ablation](../../figures/ablations/route_b_heads.png){ width=600 }

## 4. Visualizations

### Attention Maps
Attention maps in early layers show dense connectivity corresponding to computing the kernel matrix entries $K_{ij} = \phi_i^T \phi_j$.

### Convergence Trajectories
Overlaying the theoretical PCG rate $\rho = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ matches the convergence curves of the CG algorithm (as expected, since we are running CG directly).

---
**Conclusion**: The construction correctly encodes CG states in a linearly decodable form, the expected failure modes appear under ill-conditioning, and the two-head structure is necessary for the dot-product mat-vec approximation. These results validate the internal consistency of the theoretical construction.

---

## 5. Probing a Trained Transformer (NEW)

We trained a 12-layer Transformer (9.5M params, 256-dim, 4 heads) on ICL linear regression tasks via SGD (50k steps, batch 64, cosine LR schedule, RTX PRO 2000 Blackwell GPU, ~21 min). The model achieves MSE 0.013, near the noise floor ($\sigma^2 = 0.01$), confirming strong ICL performance.

### Methodology
We extracted per-layer activations at the query token position for 500 held-out test problems. For each problem, we computed the theoretical CG and GD trajectories on the dot-product kernel $K = XX^T$, and fitted linear probes (ridge regression, 80/20 train/test split) to recover these states from the model's activations.

### Results: CG vs GD Probe Recovery

| Layer | CG Probe | GD Probe | Random Control |
|-------|----------|----------|----------------|
| 1     | 0.884    | **0.907**| 0.030          |
| 2     | **0.626**| 0.539    | 0.001          |
| 3     | 0.192    | **0.248**| 0.009          |
| 5     | 0.172    | **0.370**| -0.012         |
| 8     | 0.022    | **0.168**| 0.004          |
| 12    | 0.032    | **0.156**| -0.025         |

**Mean cosine similarity**: CG = 0.184, GD = **0.298**, Random = 0.001.

![CG vs GD Probe Cosine Similarity](../../docs/figures/trained/probe_cosine_sim.png){ width=600 }

### Interpretation

**The trained model's internal states are more aligned with GD than CG.** This is consistent with von Oswald et al. (2023) and the broader GD-ICL literature. Key observations:

1. **Layer 1**: Both CG and GD probes score ~0.9. At step 1, CG and GD produce nearly identical solutions for well-conditioned problems, so this is expected.
2. **Layer 2**: CG probe (0.626) is slightly higher than GD (0.539) — the one layer where CG appears to better describe the model.
3. **Layers 3-12**: GD probe consistently dominates (0.15-0.37 vs 0.02-0.19).
4. **Random control**: Near zero throughout, confirming the probes are detecting genuine structure.

**Important caveat**: CG converges faster than GD, so at later layers the CG state variables ($\alpha_t, r_t, p_t$) have converged to a fixed point with less cross-problem variance. This makes them inherently harder to probe for. The declining CG probe similarity may partly reflect this variance reduction, not just that the model isn't doing CG. A controlled comparison on ill-conditioned problems (where CG converges more slowly) would help resolve this.

### Per-Layer Convergence

Using per-layer readout heads trained on isotropic data ($\kappa \approx 1$), the model's prediction error decreases across layers:

| Layer | Normalized Error |
|-------|-----------------|
| 1     | 1.000           |
| 3     | 0.154           |
| 6     | 0.024           |
| 9     | 0.012           |
| 12    | 0.014           |

The model achieves ~99% error reduction by layer 9, implementing an iterative refinement consistent with an optimization-based ICL mechanism. The convergence is faster than a single GD step per layer would predict for moderate $\kappa$, but the probe evidence points more toward GD than CG as the specific mechanism.

### Honest Assessment

This experiment addresses the critical gap identified in the external review (W1): **no prior experiment in this project involved a trained neural network**. The result does *not* support the CG hypothesis for this particular trained model and training distribution. The model appears to implement a GD-like algorithm, consistent with prior work. This is a genuine finding, reported honestly regardless of whether it supports the project's thesis.

**Future work to strengthen the comparison**:
- Train on tasks with varying condition numbers (mixed $\kappa$), where CG and GD rates diverge
- Compare convergence rates against CG vs GD theory curves at moderate $\kappa$ (5-50)
- Test pre-trained LLMs (GPT-2, LLaMA) for CG signatures during ICL
- Use the A-norm ($\|e_t\|_A$) for convergence comparison, as CG's theoretical guarantee is on this norm
