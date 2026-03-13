# Route B — ε-Approximation Bound for Unnormalized Mat-Vec

## Setup

Consider the two-head softmax construction for approximating the unnormalized dot-product matrix-vector product $Kv = (\Phi\Phi^T)v$ where $\Phi \in \mathbb{R}^{n \times p}$ are support features and $v \in \mathbb{R}^n$ is an arbitrary vector.

**Construction.** Given temperature parameter $\varepsilon > 0$:

- **Head 1 (Scaled Softmax):** Computes $h_1 = \text{softmax}(\varepsilon \cdot QK^T) \cdot V$ where $Q = K = \Phi$, $V = v$.
- **Head 2 (Uniform Mean):** Computes $h_2 = \mathbf{1}\mathbf{1}^T V / n = \bar{v} \cdot \mathbf{1}$, the broadcasted mean of $V$.
- **Combination:** $\widehat{Kv} = \frac{n}{\varepsilon}(h_1 - h_2)$.

## Per-Iteration Error Bound

**Proposition (Per-iter ε-bound).** For centered inputs ($\mathbf{1}^T v = 0$ and $\Phi$ column-centered), the two-head construction satisfies:

$$\frac{\|\widehat{Kv} - Kv\|_2}{\|Kv\|_2} \leq C \cdot \varepsilon \cdot \|\Phi\|_F^2 / n$$

where $C$ depends on the spectral properties of $\Phi$. Specifically, the Taylor expansion of softmax gives:

$$\text{softmax}(\varepsilon S)_{ij} = \frac{1}{n}\left(1 + \varepsilon S_{ij} - \varepsilon \bar{S}_i + O(\varepsilon^2 \|S\|_\infty^2)\right)$$

where $S = \Phi\Phi^T$ and $\bar{S}_i = \frac{1}{n}\sum_j S_{ij}$. Thus:

$$h_1 = \frac{1}{n}\sum_j v_j + \frac{\varepsilon}{n}\sum_j S_{ij} v_j - \frac{\varepsilon}{n}\bar{S}_i \sum_j v_j + O(\varepsilon^2)$$

For centered $v$ ($\sum_j v_j = 0$), the mean terms vanish and:

$$h_1 - h_2 = \frac{\varepsilon}{n}(Kv) + O(\varepsilon^2 \|S\|_\infty^2 \|v\|_1 / n)$$

Rescaling by $n/\varepsilon$ recovers $Kv$ with error $O(\varepsilon \|S\|_\infty^2 \|v\|_1)$.

**Target.** With $\varepsilon = 10^{-4}$ and $\|\Phi\|_F = O(\sqrt{np})$, the per-iteration relative error is:

$$\varepsilon_{\text{iter}} \leq 10^{-2}$$

for typical problem sizes ($n \leq 128$, $p \leq 64$).

## Cumulative Error Over CG Iterations

**Proposition (Cumulative CG error).** When using the two-head approximation for the mat-vec in each CG iteration, the total error after $t$ steps satisfies:

$$\|\alpha_t - \alpha_*\|_{K+\lambda I} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^t \|\alpha_*\|_{K+\lambda I} + O(\varepsilon_{\text{iter}} \cdot t)$$

where $\kappa = \text{cond}(K + \lambda I)$ and the $O(\varepsilon_{\text{iter}} \cdot t)$ term arises from accumulating per-step mat-vec errors.

**Proof sketch.** Each CG step with approximate mat-vec $\widetilde{Ap} = Ap + e_t$ (where $\|e_t\| \leq \varepsilon_{\text{iter}} \|Ap\|$) introduces an error that propagates linearly through the CG recurrence. By standard inexact CG analysis (Simoncini & Szyld, 2003), the convergence rate is preserved up to an additive term bounded by:

$$\sum_{s=1}^{t} \|e_s\| \cdot \prod_{j=s+1}^{t} \rho_j \leq \varepsilon_{\text{iter}} \cdot t \cdot \max_s \|Ap_s\|$$

where $\rho_j = (\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ is the per-step contraction factor.

## Centering Requirement

The construction requires centered inputs for the mean-subtraction trick to work. In practice:

1. **Support features:** Center $\Phi$ column-wise before computing the kernel. This is a preprocessing step that can be absorbed into the feature map.
2. **CG direction vectors:** The residual $r_t$ is approximately centered for well-conditioned problems (since $r_t = y - K\alpha_t$ and both terms have similar statistics).
3. **Non-centered case:** For general $v$, decompose $v = v_c + \bar{v}\mathbf{1}$ where $v_c$ is centered. Then $Kv = Kv_c + \bar{v}K\mathbf{1}$. The term $K\mathbf{1}$ (row sums) can be precomputed with an additional head or cached.

## Empirical Validation

Our implementation (`src/softmax/route_b.py`) demonstrates:

- Per-iteration operator error $\leq 10^{-2}$ for $\varepsilon = 10^{-4}$, $n = 64$, $p = 16$.
- CG convergence with approximate mat-vec tracks exact CG within the predicted $O(\varepsilon t)$ envelope.
- See `experiments/route_b_approx.py` for full results and `tests/test_route_b.py` for unit tests validating the error target.

## References

- Simoncini, V. & Szyld, D.B. (2003). "Theory of inexact Krylov subspace methods and applications to scientific computing." *SIAM J. Sci. Comput.*
- Greenbaum, A. (1997). "Iterative Methods for Solving Linear Systems." SIAM.
