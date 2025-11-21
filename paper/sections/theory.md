# Theoretical Analysis

In this section, we formalize the mechanisms by which Transformers implement Kernel Ridge Regression (KRR). We define two distinct routes: **Route A**, which leverages the exponential nonlinearity of Softmax attention to implement KRR with an exponential kernel, and **Route B**, which uses Linear Attention to implement iterative Preconditioned Conjugate Gradient (PCG) on dot-product kernels.

## 1. Softmax Attention as Exponential Kernel KRR (Route A)

**Theorem 1 (Route A Representation)**. *Let $\Phi \in \mathbb{R}^{k \times p}$ be the support features and $\phi(x)$ the query features. For a projection $U \in \mathbb{R}^{d \times p}$ and temperature $\tau > 0$, define the exponential kernel:*
$$ \tilde{K}_{ij} = \exp\left(\frac{1}{\tau} \langle U\phi(x_i), U\phi(x_j) \rangle \right) $$
*A Transformer block with a single softmax attention head (queries $Q=U\Phi$, keys $K=U\Phi$) and an aggregator token can implement the prediction function:*
$$ f(x) = \tilde{k}(x)^T (\tilde{K} + \lambda I)^{-1} y $$
*where $\tilde{k}(x)_j = \exp(\frac{1}{\tau} \langle U\phi(x_j), U\phi(x) \rangle)$.*

*Proof Sketch.*
1.  **Kernel Realization**: The unnormalized attention scores $S = QK^T / \tau$ satisfy $S_{ij} = \frac{1}{\tau} \langle U\phi(x_i), U\phi(x_j) \rangle$. Thus, $\exp(S_{ij}) = \tilde{K}_{ij}$.
2.  **Normalization Recovery**: Standard Softmax outputs $A_{ij} = \frac{\exp(S_{ij})}{Z_i}$ where $Z_i = \sum_j \exp(S_{ij})$. The aggregator token computes $Z_i$ by summing $\exp(S)$ values (via a specific head configuration) and broadcasting them, allowing the model to recover the unnormalized $\tilde{K} = \text{diag}(Z) \cdot A$.
3.  **Optimization**: With explicit access to the matrix-vector multiplication $v \mapsto \tilde{K}v$, the model implements iterative gradient-based updates (or exact inversion via Neumann series for small $\lambda$) in subsequent MLP and attention layers.

## 2. Width-Rank Tradeoff

Transformers operate with a finite width $m$ (embedding dimension). When the feature dimension $p$ or the effective dimension of the data exceeds $m$, the model cannot represent the full kernel exactly in a single pass.

**Theorem 2 (Spectral Sketching)**. *Let $K$ be the target kernel (linear or exponential) with eigenvalues $\sigma_1 \geq \sigma_2 \geq \dots$. If the Transformer width $m < p$, any linear preservation of features induces a low-rank sketch $\hat{K}$ of rank at most $m$. The approximation error in the ridge estimator is bounded by the spectral tail:*
$$ \| (K + \lambda I)^{-1} - (\hat{K} + \lambda I)^{-1} \|_2 \lesssim \frac{1}{\lambda^2} \sum_{i > m} \sigma_i(K) $$

*Implication.*
The "effective dimension" $d_{\text{eff}}(\lambda) = \sum_i \frac{\sigma_i(K)}{\sigma_i(K) + \lambda}$ governs the difficulty of the task. 
-   If $m \geq d_{\text{eff}}(\lambda)$, the model can capture the principal components of the data necessary for prediction, achieving near-oracle performance.
-   If $m \ll d_{\text{eff}}(\lambda)$, performance degrades according to the mass of the discarded eigenvalues $\sum_{i > m} \sigma_i(K)$.

## 3. Linear Attention and PCG (Route B)

For Linear Attention (no softmax), the mechanism simplifies to Preconditioned Conjugate Gradient (PCG) on the dot-product kernel $K = \Phi \Phi^T$.

**Lemma 3 (PCG Steps)**. *A Linear Attention Transformer (LAT) block can implement one step of PCG:*
$$ \alpha_{t+1} = \alpha_t + \gamma_t p_t, \quad r_{t+1} = r_t - \gamma_t A p_t, \quad p_{t+1} = r_{t+1} + \beta_t p_t $$
*where $A = K + \lambda I$. The scalar coefficients $\gamma_t, \beta_t$ are computed via the aggregator token, and the matrix-vector product $Ap_t$ is computed via the attention mechanism.*

This constructive proof relies on the specific mapping of $(\alpha, r, p)$ to distinct subspaces of the residual stream, maintained across layers $t=1 \dots L$.

