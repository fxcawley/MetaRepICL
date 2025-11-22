# Softmax Route A — Theorem (Exponential Kernel KRR)

Let Φ ∈ R^{k×p} be support features and φ(x) ∈ R^p the query features from lower layers. For U ∈ R^{d×p} and temperature τ>0, define the exponential kernel

\[ \tilde K_{ij} = \exp\!\big(\tfrac{1}{\tau}\langle U\phi(x_i), U\phi(x_j)\rangle \big),\qquad \tilde k_j(x) = \exp\!\big(\tfrac{1}{\tau}\langle U\phi(x_j), U\phi(x)\rangle \big). \]

Then a **Deep Transformer** constructed with $L$ layers of attention heads (queries Q=UΦ, keys K=UΦ) and MLP blocks can implement **Gradient Descent** steps on the kernel ridge regression objective, converging to:

\[ f_*(x) = \tilde k(x)^\top (\tilde K + \lambda I)^{-1} y. \]

A **single** softmax attention head, however, implements the normalized Nadaraya-Watson estimator:
\[ f_{NW}(x) = \frac{\sum_j \tilde k_j(x) y_j}{\sum_j \tilde k_j(x)} \]
which shares the kernel geometry but differs in prediction.

Sketch of mapping (Deep KRR):

- Unnormalized logits S = QK^T/τ yield elementwise exp(S) = \tilde K entrywise.
- Softmax normalizes rows by Z_i = ∑_j exp(S_{ij}). Two strategies:
  1) Work in the exponential-kernel space directly, interpreting softmax weights as normalized kernel smoother and folding normalization into value channels and aggregator reductions; or
  2) Recover exp(S) via a secondary channel that multiplies softmax weights by Z_i, which is computable by an aggregator token that sums exp(S_{ij}).
- Using aggregator reductions, compute necessary ridge quantities and solve via CG as in LAT→CG; the per-step mat-vec uses exp-kernel via softmax logits.

Assumptions and notes:

- The lower Φ is treated fixed. The implementation uses causal masks to prevent query→support leakage during compute.
- Numerical stability uses float32 in runtime with double-precision unit tests; λ>0 ensures conditioning.
- Approximate variants (Route B) emulate unnormalized dot-product kernels with ε per-iter operator error.

Consequences:

- Depth–accuracy follows CG rate on \(\tilde K+\lambda I\).
- Width–rank follows capacity to carry Uφ and reductions; when width<m<p, effective rank-m sketching behavior emerges.
