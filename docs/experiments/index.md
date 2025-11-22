# Experiments and Mechanistic Analysis

We present empirical evidence supporting the hypothesis that Transformers implement ICL via Route A (Softmax KRR) and Route B (PCG), using both synthetic validations and mechanistic probes.

## 1. Route A: Softmax as Exponential Kernel KRR

To validate Theorem 1, we compared the predictions of a minimal Softmax Attention layer against a ground-truth Kernel Ridge Regression oracle using the exponential kernel $K(x, x') = \exp(\langle x, x' \rangle / \tau)$.

**Results.**
As shown in Figure 1 (`figures/route_a_mvp.png`), the Softmax model's predictions align closely with the KRR oracle in terms of kernel geometry. Note that while the kernels match, a single Softmax layer acts as a normalized smoother (Nadaraya-Watson), which performs differently from the full inverse KRR at the boundaries.
-   **Operator Norm**: We measured the operator norm difference $\| \tilde{K}_{softmax} - K_{exp} \|_{op}$ on the support set. The error decreases with the "Aggregator" correction, confirming that the model recovers the unnormalized kernel geometry required for ridge regression.
-   **Temperature Sensitivity**: Ablation studies (`figures/ablations/route_a_temp.png`) demonstrate that this alignment holds across a range of temperatures $\tau$, whereas the normalized softmax baseline diverges.

## 2. Route B: Preconditioned Conjugate Gradient

For linear attention, we hypothesized an iterative PCG mechanism. We validated this via failure mode analysis and internal state probing.

### 2.1 Failure Modes and Preconditioning
Standard Gradient Descent (and standard CG) stalls on ill-conditioned problems where the condition number $\kappa(K)$ is large.
-   **Ill-Conditioned Stall**: We constructed synthetic datasets with $\kappa \in [10^0, 10^3]$. We observed that standard Transformer optimization slows down on high-$\kappa$ tasks, matching the theoretical CG rate.
-   **Preconditioning Recovery**: We demonstrated that the model recovers convergence speed consistent with diagonal preconditioning $P^{-1} \approx (\text{diag}(K) + \lambda I)^{-1}$. Figure 2 (`figures/failure_modes/ill_conditioned_cg.png`) shows the recovery of convergence rates when using the token-wise scaling approximation of the preconditioner.

### 2.2 Probing Optimization States
We trained linear probes to recover the theoretical CG variables—search direction $p_t$, residual $r_t$, and solution $\alpha_t$—from the residual stream at each layer.
-   **High Fidelity Recovery**: Probes recover these states with cosine similarity $> 0.9$ in middle-to-late layers (`figures/probes/cosine_sim_layer.png`), indicating the model explicitly instantiates these algorithmic variables.
-   **Specificity**: Random control probes fail to recover these directions, confirming the signal is non-trivial.

## 3. Ablation Studies

To confirm the causal necessity of the proposed components, we performed targeted ablations.

**Head Sharing (Route B).**
Theorem 3 requires two heads to implement the update $v \leftarrow v - \text{mean}(v)$. We ablated the second "aggregation" head.
-   **Impact**: As shown in `figures/ablations/route_b_heads.png`, removing the aggregation head degrades the approximation of the dot-product mat-vec, causing the effective error to plateau significantly higher than the full two-head construction. This confirms the "Mean Subtraction" role of the secondary head.

## 4. Width-Rank Tradeoff

Finally, we validated the spectral sketching limits (Theorem 2). We varied the Transformer width $m$ while keeping the data dimension $d_{eff}$ fixed.
-   **Spectral Tail**: The prediction error follows the predicted spectral tail curve (`figures/width_rank_curve.png`). Performance collapses precisely when $m < d_{eff}$, consistent with the interpretation that the Transformer performs an implicit low-rank sketch of the kernel.

