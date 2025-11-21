# Background and Related Work

## 1. In-Context Learning as Optimization

In-Context Learning (ICL) refers to the ability of large language models (LLMs) to perform tasks defined by a sequence of examples (prompts) without explicit weight updates (Brown et al., 2020). While early interpretations focused on Bayesian inference (Xie et al., 2021), a growing body of work characterizes ICL as an implicit optimization process occurring within the forward pass.

### 1.1 Gradient Descent and Regression
Seminal work by von Oswald et al. (2023) and Dai et al. (2023) demonstrated that self-attention layers can implement steps of Gradient Descent (GD). Specifically, for linear regression tasks $y = Wx$, a single linear attention head can effectively compute a GD step $W_{t+1} \leftarrow W_t - \eta \nabla L$, where the "gradients" are computed via the correlation between queries and keys in the prompt context.

However, standard GD converges slowly on ill-conditioned problems (linear convergence rate dependent on condition number $\kappa$). Transformers are observed to solve ICL tasks much more efficiently than simple GD would predict, often reaching high accuracy in few layers (iterations), suggesting a more sophisticated optimization algorithm is at play.

### 1.2 Meta-Learning and Representations
ICL is often viewed through the lens of meta-learning, where the pre-training phase learns a learning algorithm (meta-learning) that is executed at inference time (learning). A crucial component of this is the learned representation space. Unlike "pure" optimization on raw inputs, Transformers operate on latent representations.

Our work builds on the hypothesis that Transformers learn **meta-representations** that facilitate efficient optimization. Specifically, we investigate the claim that they align these representations to enable **Kernel Ridge Regression (KRR)** via two distinct mechanisms: an explicit Preconditioned Conjugate Gradient (PCG) descent in linear attention layers, and a closed-form exponential kernel solution in softmax attention layers.

## 2. Kernel Methods and Preconditioning

### 2.1 Kernel Ridge Regression (KRR)
KRR solves the optimization problem $\min_w \sum_i (y_i - w^T \phi(x_i))^2 + \lambda \|w\|^2$ via the dual solution $\alpha = (K + \lambda I)^{-1} y$. This solution requires inverting the kernel matrix $K_{ij} = \phi(x_i)^T \phi(x_j)$.

### 2.2 Conjugate Gradient (CG) and Preconditioning
Solving linear systems $Ax=b$ (where $A = K + \lambda I$) can be done iteratively via the Conjugate Gradient method. CG converges in at most $n$ steps for $n$-dimensional systems, but its practical convergence rate depends on the distribution of eigenvalues of $A$.
Specifically, the error at step $t$ is bounded by:
$$ \|x_t - x_*\|_A \leq 2 \left( \frac{\sqrt{\kappa(A)} - 1}{\sqrt{\kappa(A)} + 1} \right)^t \|x_0 - x_*\|_A $$
where $\kappa(A)$ is the condition number.

When $\kappa$ is large, CG slows down. **Preconditioning** transforms the system to $P^{-1}Ax = P^{-1}b$ using a symmetric positive-definite $P \approx A$, reducing the effective condition number $\kappa(P^{-1}A)$. We propose that Transformers implement a specific form of diagonal preconditioning to handle ill-conditioned data distributions efficiently.

## 3. Transformers as Mesa-Optimizers
This work connects the mechanistic interpretability of Transformers (Olsson et al., 2022; Elhage et al., 2021) with the optimization perspective. While previous studies have identified "induction heads" (copying mechanisms), we identify **"optimization heads"**—specialized attention heads that perform specific arithmetic operations (mat-vec multiplication, aggregations) corresponding to steps of the PCG algorithm (Akyürek et al., 2024; Mahankali et al., 2023).

We distinguish between two regimes:
1.  **Route A (Softmax)**: Implementing the KRR solution directly via the attention matrix as a kernel $K_{exp}(x, x') = \exp(\langle x, x' \rangle / \tau)$.
2.  **Route B (Linear)**: Implementing iterative updates matching PCG.

By formally mapping the architectural components of the Transformer (Attention, MLP, LayerNorm) to the algebraic steps of these algorithms, we provide a rigorous "compiler" view of ICL.

