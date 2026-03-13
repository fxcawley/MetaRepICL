# Experiments and Mechanistic Analysis

We present numerical validation of our theoretical constructions, verifying that the proposed mathematical mappings are self-consistent. All experiments below operate on hand-built constructions (analytical formulas and direct matrix computations), not on transformers trained via SGD. Validation on trained models is future work (see [status.md](../../docs/status.md)).

## 1. Route A: Softmax as Exponential Kernel KRR

To validate Theorem 1, we compared the predictions of a minimal Softmax Attention layer against a ground-truth Kernel Ridge Regression oracle using the exponential kernel $K(x, x') = \exp(\langle x, x' \rangle / \tau)$.

**Results.**
As shown in Figure 1 (`figures/route_a_mvp.png`), Conjugate Gradient on the softmax-reconstructed exponential kernel converges to the KRR oracle in $\sim$10 iterations (left panel), while gradient descent on the same kernel converges significantly slower (right panel). This validates the theory's claim (Corollary A) that CG applies to Route A with $K \to \tilde{K}$, achieving the CG convergence rate $\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t$ rather than the GD rate $\left(\frac{\kappa-1}{\kappa+1}\right)^t$.
-   **Operator Norm**: We measured the operator norm difference $\| \tilde{K}_{softmax} - K_{exp} \|_{op}$ on the support set. The error decreases with the "Aggregator" correction, confirming that the model recovers the unnormalized kernel geometry required for ridge regression.
-   **Temperature Sensitivity**: Ablation studies (`figures/ablations/route_a_temp.png`) demonstrate that this alignment holds across a range of temperatures $\tau$, whereas the normalized softmax baseline diverges.

## 2. Route B: Preconditioned Conjugate Gradient

For linear attention, we hypothesized an iterative PCG mechanism. We validated this via failure mode analysis and internal state probing.

### 2.1 Failure Modes and Preconditioning
Standard Gradient Descent (and standard CG) stalls on ill-conditioned problems where the condition number $\kappa(K)$ is large.
-   **Ill-Conditioned Stall**: We constructed synthetic datasets with $\kappa \in [10^0, 10^3]$. The CG algorithm slows down on high-$\kappa$ problems, matching the theoretical CG rate.
-   **Preconditioning Recovery**: Diagonal preconditioning $P^{-1} \approx (\text{diag}(K) + \lambda I)^{-1}$ restores convergence speed. Figure 2 (`figures/failure_modes/ill_conditioned_cg.png`) shows the recovery of convergence rates when using the token-wise scaling approximation. *Note: The preconditioned variant solves a slightly different problem (the effective regularizer changes), making this comparison somewhat apples-to-oranges. See `ill_conditioned.py` comments for details.*

### 2.2 Probing Optimization States
We trained linear probes to recover the theoretical CG variables—search direction $p_t$, residual $r_t$, and solution $\alpha_t$—from the *constructed* (not trained) model's residual stream at each layer.

**Important caveat**: These probes operate on our hand-built construction, where "activations" are a known random linear projection $W_{\text{true}} \cdot z$ of the true CG states $z = [\alpha; r; p]$. The probe recovers $z$ by inverting this linear map — a standard linear algebra operation, not a finding about trained transformers. The high cosine similarity validates that the construction correctly encodes CG states, but says nothing about whether trained models do the same. Probing trained models is the critical next step (see [status.md](../../docs/status.md)).

-   **High Fidelity Recovery**: Probes recover these states with cosine similarity $> 0.9$ from the constructed embedding (`figures/probes/cosine_sim_layer.png`), confirming the construction correctly encodes CG variables in a linearly decodable form.
-   **Specificity**: Random control probes (noise inputs) fail to recover these directions, confirming the signal depends on the constructed CG states rather than being an artifact of the probe fitting procedure.

## 3. Ablation Studies

To test the structural necessity of specific components in our construction, we performed targeted ablations on the analytical construction (not on trained models).

**Head Sharing (Route B).**
Theorem 3 requires two heads to implement the update $v \leftarrow v - \text{mean}(v)$. We ablated the second "aggregation" head.
-   **Impact**: As shown in `figures/ablations/route_b_heads.png`, removing the aggregation head degrades the approximation of the dot-product mat-vec, causing the effective error to plateau significantly higher than the full two-head construction. This confirms the "Mean Subtraction" role of the secondary head.

## 4. Width-Rank Tradeoff

Finally, we validated the spectral sketching limits (Theorem 2). We varied the Transformer width $m$ while keeping the data dimension $d_{eff}$ fixed.
-   **Spectral Tail**: The prediction error follows the predicted spectral tail curve (`figures/width_rank_curve.png`). Performance collapses precisely when $m < d_{eff}$, consistent with the interpretation that the Transformer performs an implicit low-rank sketch of the kernel.

