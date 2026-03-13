# ICL on Real-World Data: Kernel Smoother vs Ridge

We extended our experiments to real-world text classification (AG News, TREC) using Bag-of-Words features. We compared Ridge regression with a single-layer Softmax Attention kernel smoother (Nadaraya-Watson estimator with exponential kernel).

**Important note**: The "Softmax Attention" model here is a hand-coded kernel smoother: `scores = Q @ K.T / tau`, `softmax(scores)`, `weights @ V`. It has zero learned parameters (no W_Q, W_K, W_V projections, no MLP, no LayerNorm, no residual connections, no training). It is a classical Nadaraya-Watson estimator, not a transformer. We use it to test whether the kernel smoother interpretation of Route A produces reasonable predictions on real data.

## Results

### AG News (Topic Classification)

| Support ($k$) | Ridge MSE | Kernel Smoother MSE |
| :--- | :--- | :--- |
| 8 | 1.49 | 1.44 |
| 16 | 1.44 | 1.43 |
| 32 | 1.33 | 1.28 |
| 64 | 1.18 | 1.43 |
| 128 | 0.95 | 1.05 |

**Analysis**:
- For small $k$ (8-32), the **Kernel Smoother (Softmax Attention)** slightly outperforms Ridge. This aligns with the "Kernel Smoother" view: Softmax attention acts as a weighted nearest neighbor classifier, which is robust in low-data regimes.
- For larger $k$ (64-128), **Ridge** overtakes. Ridge (Linear Regression) can learn the global linear boundary better as data increases, whereas the kernel smoother (without MLP layers) is limited to convex combinations of labels (effectively a Parzen window estimator).
- This is consistent with the theoretical intuition: Softmax Attention is akin to a Kernel method (Route A), while Linear Attention/Ridge corresponds to solving the linear system.

### Visualization: Attention Maps
We generated attention maps for the kernel smoother model on AG News.
*(See `figures/attention/attn_seed123.png`)*

The attention weights show which support examples the model focuses on for a given query. In a successful case, the model attends highly to support examples with the same label as the query (or semantically similar text).

## Conclusion
The softmax kernel smoother (our Route A analytical construction) performs comparably to Ridge in few-shot settings but saturates differently. This is consistent with the theoretical interpretation of softmax attention as an exponential kernel method. However, this experiment involves no learned parameters and does not validate that trained transformers implement this mechanism — it only shows the kernel smoother interpretation produces reasonable predictions on real data.

