# Silent Failure of Softmax Attention as a Kernel Smoother

## Abstract

We study the numerical behavior of softmax attention interpreted as an exponential kernel smoother, identifying two failure modes that are invisible to standard loss metrics. When feature dimension is large, inner product concentration causes softmax weights to become near-uniform, collapsing rank correlation to 0.22 while RMSE degrades only ~30%. When features are ill-conditioned, the exponential kernel blows up numerically, but softmax normalization accidentally masks the failure -- producing stable predictions (RMSE=2.8) that are systematically misordered (rank correlation=0.24). We call these "silent failures" because surface metrics suggest the model is working. We show that temperature is the critical parameter controlling the transition, with a shrinking safe zone as dimension increases. Linear attention (dot-product kernel) is immune to both failure modes. Our key finding: **softmax attention is most useful as a kernel smoother exactly where the oracle kernel method is least needed**.

## 1. Setup

Softmax attention computes a Nadaraya-Watson kernel smoother with an exponential kernel. Given support features $\Phi \in \mathbb{R}^{k \times p}$, query feature $\phi_q \in \mathbb{R}^p$, labels $y \in \mathbb{R}^k$, and temperature $\tau > 0$:

$$f_{\text{softmax}}(x_q) = \sum_{i=1}^{k} \frac{\exp(\langle \phi_i, \phi_q \rangle / \tau)}{\sum_{j=1}^{k} \exp(\langle \phi_j, \phi_q \rangle / \tau)} \, y_i$$

The oracle exponential kernel ridge regression (KRR) predictor is:

$$f_{\text{oracle}}(x_q) = \tilde{k}(x_q)^\top (\tilde{K} + \lambda I)^{-1} y, \quad \tilde{K}_{ij} = \exp(\langle \phi_i, \phi_j \rangle / \tau)$$

We compare four methods:

| Method | Description |
|--------|-------------|
| **Oracle (Exp KRR)** | Exact $(\tilde{K} + \lambda I)^{-1} y$ |
| **Softmax Attention** | Nadaraya-Watson with exponential kernel |
| **Z-Recovered KRR** | Oracle KRR using kernel reconstructed from softmax + normalization constant |
| **Linear Kernel CG** | Exact KRR with $K_{ij} = \langle\phi_i, \phi_j\rangle$ |

We sweep across four axes: dimension $p$, temperature $\tau$, condition number $\kappa$ of the feature covariance, and context length $k$.

**Metrics**: RMSE alone is insufficient to detect silent failures. We use rank correlation (Spearman) to measure whether predictions preserve the ordering of ground truth values. A model with low RMSE but low rank correlation is making stable but systematically wrong predictions.

## 2. Silent Failure Mode 1: High-Dimensional Concentration

When $p$ is large, inner products $\langle \phi_i, \phi_j \rangle$ concentrate around a common value by the law of large numbers. This means $\exp(\langle \phi_i, \phi_j \rangle / \tau) \approx c$ for all pairs, so softmax weights become **nearly uniform** and the model predicts approximately $\bar{y}$ for every query.

| p | Softmax RMSE | Rank Correlation | Z (normalization) |
|---|-------------|-----------------|-------------------|
| 2 | 0.6 | 0.92 | 280 |
| 8 | 1.4 | 0.83 | 1,500 |
| 16 | 2.4 | 0.63 | 73,000 |
| 32 | 4.4 | 0.48 | $10^5$ |
| 64 | 9.3 | 0.18 | $10^{10}$ |

At $p=64$, RMSE is only ~30% worse than the oracle, but rank correlation = 0.18 means predictions are in nearly random order. The linear kernel CG maintains rank correlation > 0.999 at all dimensions.

## 3. Silent Failure Mode 2: Ill-Conditioned Features

When the feature covariance has extreme eigenvalue spread ($\kappa \gg 1$), the exponential kernel amplifies this catastrophically. At $\kappa = 500$:

- **Oracle KRR**: RMSE = $5 \times 10^{12}$ (kernel matrix is numerically singular)
- **Softmax**: RMSE = 2.8 (looks reasonable)
- **Softmax rank correlation**: 0.24 (predictions in wrong order)

Softmax normalization accidentally acts as extreme regularization, flattening the kernel into something numerically stable but informationally useless. Linear kernel CG: RMSE = 0.045, rank correlation = 0.999.

## 4. Temperature as the Critical Parameter

Temperature $\tau$ controls the transition between the two failure modes:

- **Low $\tau$ (< 0.1)**: The exponential kernel blows up. Softmax normalization stabilizes predictions but reduces them to approximately 1-nearest-neighbor.
- **Moderate $\tau$ (0.5 -- 2.0)**: Sweet spot. Softmax and oracle roughly agree. Rank correlation ~ 0.7.
- **High $\tau$ (> 10)**: $\exp(\cdot/\tau) \approx 1 + \cdot/\tau$. The exponential kernel linearizes; both methods converge but lose nonlinear kernel advantages.

**Key insight**: Softmax is most useful (stable, discriminative) exactly where the oracle is least needed (moderate $\tau$, low $p$, well-conditioned). Where the kernel really matters (extreme $\tau$, high $p$, ill-conditioned), softmax normalization destroys the signal.

The "safe zone" for $\tau$ narrows as dimension increases. In trained transformers, this corresponds to learned attention temperatures -- models that learn inappropriate temperatures will fail silently.

## 5. Practical Implications

1. **Softmax ICL will fail silently on high-dimensional regression tasks.** Surface metrics (RMSE, loss) may not catch this. Rank-based metrics are essential.

2. **Ill-conditioned problems break exponential-kernel KRR catastrophically**, but softmax normalization masks the failure. The model appears to work but has lost sensitivity to feature directions.

3. **Linear attention (dot-product kernel) is immune** to both failure modes for linear regression tasks.

4. **Temperature tuning is critical.** The safe zone narrows with dimension. In real transformers, attention temperature is learned and may enter failure regions.

5. **Detecting silent failure requires more than RMSE.** A model with RMSE=2.8 and rank correlation=0.24 is worse than one with RMSE=5.0 and rank correlation=0.95.

## 6. Scope and Limitations

This analysis compares analytical formulas (exponential-kernel KRR vs. Nadaraya-Watson estimator), not trained transformers. It characterizes the mathematical properties of softmax attention interpreted as a kernel smoother. Whether trained transformers exhibit these failure modes at the predicted thresholds is an open question. All experiments are numerical simulations on synthetic data.

## Reproducibility

All results generated by `experiments/silent_failure.py`. Requires only NumPy, SciPy, and Matplotlib. Runs in < 2 minutes on CPU.
