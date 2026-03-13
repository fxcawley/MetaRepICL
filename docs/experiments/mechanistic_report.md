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
**Conclusion**: The construction correctly encodes CG states in a linearly decodable form, the expected failure modes appear under ill-conditioning, and the two-head structure is necessary for the dot-product mat-vec approximation. These results validate the internal consistency of the theoretical construction. They do *not* constitute evidence that trained transformers use PCG — that requires probing trained models, which is future work.
