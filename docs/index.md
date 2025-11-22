# MetaRep: In-Context Learning as Meta-Representation Alignment

**Can a Transformer simulate Gradient Descent during In-Context Learning?**
Our work suggests the architecture is capable of implementing efficient optimization algorithms: **Preconditioned Conjugate Gradient (PCG)** or **Exponential Kernel Ridge Regression (KRR)**.

This project, **MetaRep**, formalizes the constructive proofs and empirically validates the **expressivity** of the Transformer architecture to implement these algorithms.

*Note: This repository focuses on mechanistic construction and capacity. Training Transformers to discover these algorithms from scratch is the subject of future work.*

[View the Worked Demo](demo.md){ .md-button .md-button--primary }

---

## The Key Idea

Transformers map input tokens $x$ to a latent space $\phi(x)$. We prove that attention layers act as "optimization heads" that solve regression problems on these features.

### Route A: Softmax Attention = Exponential Kernel KRR
Softmax attention naturally implements the Nadaraya-Watson estimator. We show that with a specific temperature scaling and aggregation, it implements **Kernel Ridge Regression** with the kernel:
$$ K(x, x') = \exp\left(\frac{\langle \phi(x), \phi(x') \rangle}{\tau}\right) $$

### Route B: Linear Attention = Preconditioned CG
Linear attention allows for iterative updates. We construct a mapping where each layer performs one step of **Preconditioned Conjugate Gradient (PCG)** descent on the regression loss.

---

## Key Results

### 1. Softmax KRR Alignment
The Transformer's attention mechanism induces a kernel that aligns with the theoretical exponential kernel (operator norm difference $< 10^{-8}$).
- **Green vs Blue**: A deep Transformer constructed to run Gradient Descent (Green) successfully recovers the Oracle KRR solution (Blue).
- **Orange**: A single layer acts as a Nadaraya-Watson smoother, sharing the kernel geometry but not the optimization path.

![Route A MVP](figures/route_a_mvp.png){ width=600 }

### 2. Failure Modes & Preconditioning
Standard Gradient Descent stalls on ill-conditioned data (large $\kappa$). Our constructed PCG model stalls too, but recovers when we introduce our proposed diagonal preconditioner.

![CG Failure Modes](figures/failure_modes/ill_conditioned_cg.png){ width=600 }

### 3. Low-Rank Sketching
When the model width $m$ is smaller than the data dimension, the Transformer acts as a **low-rank sketch** of the kernel. Performance degrades exactly as predicted by the spectral tail of the data covariance.

![Width Rank Curve](figures/sweeps/width_rank_curve.png){ width=600 }

---

## Walkthrough for Reviewers

1.  **Demo**: Start with the [Worked Demo](demo.md) to see the project in action.
2.  **Theory**: Check the [Theory Overview](theory/index.md) for the proofs.
3.  **Evidence**: See the [Experiments Summary](experiments/index.md) for empirical validation.
4.  **Code**: Run the [Reproducibility Script](reproducibility.md) to generate these figures yourself.

## Project Status
We have validated the core mechanisms on synthetic data. Current work focuses on scaling these probes to real Language Models (LLaMA-2, etc.) to see if this "Mesa-Optimizer" behavior emerges in the wild.

[Read our Honest Interpretations and Future Work](status.md){ .md-button }
