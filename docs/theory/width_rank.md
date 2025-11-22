# Width–Rank Theorem — Statement and Implications

Setup: Let Φ ∈ R^{k×p} be support features from lower layers. A depth-d transformer block above Φ has width m channels available to carry feature-aligned computations. Consider kernel ridge regression with kernel K=ΦΦ^T (or the Route-A exponential kernel \(\tilde K\)).

Theorem (Width–rank sketching): If m ≥ p + c (channels for CG state), then there exists a parameterization realizing t CG steps on KRR exactly (up to numeric precision) under the masks described in S5. If m < p, any computation that linearly preserves Φ across layers induces an effective rank-m sketch \(\hat K\) with spectral approximation error controlled by the tail of K:

\[ \| K - \hat K \|_2 \leq \sum_{i>m} \sigma_i(K), \quad \text{and} \quad \| (K+\lambda I)^{-1} - (\hat K+\lambda I)^{-1} \|_2 \lesssim \frac{1}{\lambda^2} \sum_{i>m} \sigma_i(K), \]

which yields a query prediction error bound for any x with feature vector φ(x):

\[ |k(x)^\top [(K+\lambda I)^{-1} - (\hat K+\lambda I)^{-1}] y| \;\le\; \|k(x)\|_2 \cdot \|y\|_2 \cdot \|(K+\lambda I)^{-1} - (\hat K+\lambda I)^{-1}\|_2. \]

Moreover, writing the effective dimension \( d_\text{eff}(\lambda) = \mathrm{tr}(K (K+\lambda I)^{-1}) = \sum_i \frac{\sigma_i(K)}{\sigma_i(K)+\lambda} \), width m controls approximation via the spectral tail beyond m: when \(m \ge d_\text{eff}(\lambda)\), the induced error is small; when \(m \ll d_\text{eff}(\lambda)\), prediction degrades in proportion to the tail mass \(\sum_{i>m} \sigma_i(K)\).

Implications:

- Depth–accuracy: unchanged from CG rate provided the mat-vec uses K (or \(\tilde K\)); with sketch \(\hat K\), an additional bias arises from \(K\to\hat K\) perturbation.
- Diagnostics: report both prediction error and \(d_\text{eff}(\lambda)\); sweeping m should track the spectral tail predictions.
- Softmax Route A: identical statements hold with K replaced by \(\tilde K\) and σ_i replaced by the spectrum of \(\tilde K\).

Notes: Constants can be tightened with standard resolvent perturbation bounds; operator-norm statements can be strengthened to relative error bounds under eigengap assumptions.
