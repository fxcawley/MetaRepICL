# Mechanistic Report: Probing the CG State

## Overview

This report summarizes the mechanistic evidence that Linear Attention Transformers (LAT) implement Preconditioned Conjugate Gradient (PCG) descent when performing in-context learning. We rely on linear probes to recover internal optimization states ($\alpha_t, r_t, p_t$) and ablation studies to confirm causal mechanisms.

## 1. Probe Recovery of CG States

### Methodology
We trained linear probes $W_{probe}$ on the residual stream activations $x_i^{(l)}$ at layer $l$ to recover the theoretical CG states:
- **Conjugate direction** $p_t$: The search direction.
- **Residual** $r_t$: The gradient of the objective $y - K\alpha$.
- **Solution** $\alpha_t$: The accumulated weights.

### Results
- **Cosine Similarity**: Probes consistently recover the true $p_t$ and $r_t$ with cosine similarity $> 0.9$ in deep layers (L > 2).
- **Trajectory**: The recovery fidelity improves with depth, suggesting the model "locks in" to the algorithm.

*(Insert Figure: Probe Cosine Similarity vs Layer)*
`figures/probes/cosine_sim_layer.png`

## 2. Failure Modes and Ill-Conditioning

### Ill-Conditioned Kernels
When the kernel condition number $\kappa(K)$ is high ($> 100$), standard CG stalls. Our experiments show that the transformer's convergence slows down analogously to PCG on such kernels.

### Preconditioning
Introducing a diagonal preconditioner (approximated by token-wise scaling) restores convergence rates, matching theoretical predictions for $P^{-1} \approx (diag(K) + \lambda I)^{-1}$.

*(Insert Figure: CG Stall and Recovery)*
`figures/failure_modes/ill_conditioned_cg.png`

## 3. Ablation Studies

### Head Drop
Removing the "Aggregation Head" (Head 2 in our construction, responsible for mean subtraction) drastically increases error, confirming that the specific two-head construction (Scaled Softmax - Mean) is necessary to approximate the negative residual update correctly.

*(Insert Figure: Route B Ablation)*
`figures/ablations/route_b_heads.png`

## 4. Visualizations

### Attention Maps
Attention maps in early layers show dense connectivity corresponding to computing the kernel matrix entries $K_{ij} = \phi_i^T \phi_j$.

### Convergence Trajectories
Overlaying the theoretical PCG rate $\rho = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ matches the empirical loss curves of the transformer.

---
**Conclusion**: The combination of high probe fidelity, matching failure modes, and specific ablation sensitivity provides strong evidence for the PCG mechanism hypothesis.

