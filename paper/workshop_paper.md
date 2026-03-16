# Limits of Algorithm Identification in In-Context Linear Regression

## Abstract

In-context learning (ICL) in transformers has been linked to gradient descent (GD), but trained models converge faster than GD predicts. We provide a systematic algorithm identification study for ICL, comparing trained transformers against six named iterative algorithms---vanilla GD, conjugate gradients (CG), preconditioned GD (Jacobi), heavy ball, Chebyshev iteration, and preconditioned CG---with bootstrap confidence intervals and condition-number stratification. Our main findings: **(1)** trained transformers dramatically outperform GD (R^2 gap = 0.20, p < 0.01), consistent with second-order convergence; **(2)** CG, preconditioned CG, and preconditioned GD are statistically indistinguishable as descriptions of the model (gap 0.004, CI 0.017), suggesting that specific algorithm identification may require larger experimental scales; **(3)** linear probes find internal representations more GD-like than CG-like despite CG-like convergence rates, a suggestive mismatch that requires further investigation with nonlinear probes. Additionally, we characterize silent failure modes of softmax attention as an approximate kernel regressor, identifying regimes where predictions appear plausible but rank ordering is destroyed. Our results suggest that the correct characterization is "CG-convergence-class optimization" rather than any single named algorithm, and that standard observational comparisons face fundamental identifiability limits at current experimental scales.


## 1. Introduction

Transformers perform in-context learning (ICL): given a sequence of input-output examples followed by a query, they produce predictions without weight updates. Understanding *what algorithm* the forward pass implements is a central question in mechanistic interpretability.

**Wave 1: ICL as gradient descent.** Von Oswald et al. (2023) showed that linear attention layers can implement GD steps, and Akyurek et al. (2023) independently found ridge-regression-like solutions in trained models. Mahankali et al. (2023) proved that one GD step is optimal for single-layer linear attention. These results established the "ICL-as-optimization" framework.

**Wave 2: ICL is better than GD.** Fu et al. (2023) observed that trained transformers achieve Newton-like (second-order) convergence rates, far exceeding what GD can explain. Ahn et al. (2024) addressed this by proving that transformers learn *preconditioned* GD with a covariance-derived preconditioner, published at NeurIPS 2024.

**The specificity problem.** Prior work typically compares two algorithms (GD vs. one alternative) and declares a winner. But how robust is this identification? If we test against a broader set of candidates, does the specific claim survive?

**Our contribution.** We test six named algorithms simultaneously with statistical rigor:

1. **CG-class best fits** (Section 3.1): The model's per-problem predictions correlate strongly (R^2 = 0.92) with CG, preconditioned CG, and preconditioned GD, and poorly with vanilla GD (R^2 = 0.72). The separation is large and significant.

2. **Specific algorithm not identifiable** (Section 3.2): The top three algorithms (CG, Precond CG, Precond GD) have R^2 values within 0.004 of each other, with bootstrap 95% CI of 0.017. Ahn et al.'s "preconditioned GD" is one member of this indistinguishable set.

3. **Probe-behavior mismatch** (Section 3.3): Linear probes for algorithm state variables find GD-like representations despite CG-like convergence---a suggestive mismatch that requires nonlinear probes and stronger controls to interpret.

4. **Silent failure characterization** (Section 4): We identify regimes where softmax attention produces plausible predictions with destroyed rank ordering, a failure mode invisible to standard RMSE evaluation.


## 2. Method

### 2.1 Experimental Setup

We train GPT-style transformers (Garg et al., 2022) on in-context linear regression. Each task samples $\mathbf{w} \sim \mathcal{N}(0, I/p)$, features $\mathbf{x}_i \sim \mathcal{N}(0, \Sigma)$ where $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_p)$ with $\sigma_j$ geometrically spaced from $\kappa$ to 1, and $y_i = \mathbf{x}_i^\top \mathbf{w} + \varepsilon_i$. The model receives $(x_1, y_1, \ldots, x_n, y_n, x_q, 0)$ and predicts $y_q$.

**Training.** We train on a *mixture* of condition numbers $\kappa \in \{1, 10, 50, 100, 500\}$, randomly sampling $\kappa$ per batch. This ensures the model encounters both well-conditioned and ill-conditioned problems. We use AdamW with cosine schedule (3e-4 peak LR, 1000 warmup steps).

**Two scales:**

| | Small | Large |
|---|---|---|
| Features $p$ | 10 | 20 |
| Support $n$ | 20 | 40 |
| Layers | 12 | 24 |
| $d_\text{model}$ | 256 | 256 |
| Parameters | 9.5M | 19M |
| Training steps | 50K | 30K |

### 2.2 Algorithm Identification Framework

For each test problem, we run six reference algorithms in **feature space**, solving $(X^\top X + \lambda I)\mathbf{w} = X^\top \mathbf{y}$:

1. **Vanilla GD**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta (B\mathbf{w}_t - \mathbf{b})$, optimal fixed $\eta = 2/(\lambda_\text{max} + \lambda_\text{min})$.

2. **Conjugate Gradients (CG)**: Standard CG with $\mathbf{w}_0 = 0$.

3. **Preconditioned GD (Jacobi)**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta M^{-1}(B\mathbf{w}_t - \mathbf{b})$ where $M = \text{diag}(B)$. Step size computed via $M^{-1/2} B M^{-1/2}$ similarity transform (required since $M^{-1}B$ is not symmetric).

4. **Heavy Ball**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla f + \beta(\mathbf{w}_t - \mathbf{w}_{t-1})$ with damped momentum ($\beta \leq 0.9$) for numerical stability at high $\kappa$.

5. **Chebyshev iteration**: Three-term recurrence with predetermined step sizes from eigenvalue bounds. No inner products required.

6. **Preconditioned CG (Jacobi)**: CG with $z = M^{-1}r$ substitution.

Each algorithm runs for $L$ steps (matching the model's layer count). At step $\ell$, we compute the prediction $\hat{y}_q^{(\ell)} = \mathbf{x}_q^\top \mathbf{w}_\ell$.

**Per-layer readout heads.** We train independent linear readout heads from each layer's query-position activation to $y_q$, following the methodology of prior probing work.

### 2.3 Metrics

**Per-problem R^2.** For each layer $\ell$ and algorithm $a$, we collect pairs $\{(\hat{y}_q^{\text{model},\ell}_i, \hat{y}_q^{a,\ell}_i)\}_{i=1}^N$ across $N$ test problems. R^2 measures how well algorithm $a$'s per-problem predictions explain the model's per-problem outputs---capturing problem-specific behavior, not just aggregate MSE.

**MSE profile distance.** $d(a) = \sqrt{\frac{1}{L}\sum_{\ell=1}^L (\log_{10} \text{MSE}_\ell^\text{model} - \log_{10} \text{MSE}_\ell^a)^2}$. This measures similarity of the convergence *shape* across layers.

**Bootstrap 95% CIs.** All R^2 values include 95% bootstrap confidence intervals (200 resamples), enabling formal distinguishability testing.

**Condition-number stratification.** All metrics are computed separately per $\kappa$. At low $\kappa$, all algorithms converge quickly (providing a sanity check). At high $\kappa$, algorithms diverge maximally, providing the most discriminative signal.


## 3. Results

### 3.1 The Model Best Fits CG-Class Optimization

Table 1 shows kappa-weighted mean R^2 (weighting higher $\kappa$ more, since that's where algorithms differ most):

| Algorithm | Weighted R^2 | 95% CI |
|---|---|---|
| Precond CG | **0.922** | $\pm$ 0.008 |
| CG | 0.918 | $\pm$ 0.009 |
| Precond GD | 0.910 | $\pm$ 0.011 |
| Chebyshev | 0.780 | $\pm$ 0.042 |
| GD | 0.721 | $\pm$ 0.060 |
| Heavy Ball | 0.581 | $\pm$ 0.046 |

The CG-class algorithms (CG, Precond CG, Precond GD) all achieve R^2 > 0.90. Vanilla GD is clearly separated at 0.72. The model converges dramatically faster than GD, consistent with Fu et al. (2023).

At $\kappa = 100$: CG R^2 = 0.91, Precond CG R^2 = 0.92, GD R^2 = 0.71. The gap between the CG family and GD (0.20) is large, consistent, and significant across both experimental scales.

### 3.2 Specific Algorithm Identification Fails

The gap between the top two algorithms (Precond CG and CG) is **0.004**, with a combined 95% CI of **0.017**. This gap is well within noise. Moreover:

- CG wins at $\kappa = 1$ and $\kappa = 500$
- Precond CG wins at $\kappa = 50$ and $\kappa = 100$
- The MSE profile distance metric disagrees with R^2 on which is best at several $\kappa$ values

Preconditioned GD---the algorithm identified by Ahn et al. (2024)---is also in this indistinguishable cluster (R^2 = 0.91). Our finding does not contradict Ahn et al.; rather, it shows their identification is one member of a class. The claim "preconditioned GD" is no more supported than "CG" or "preconditioned CG" at these scales.

**Why can't we distinguish?** With $p = 20$ features and 24 layers, CG-class methods converge in $\leq p = 20$ steps. Layers 21--24 are past convergence and carry no discriminative signal. All converging algorithms approach the same ridge solution, making late-layer R^2 uninformatively high. Scaling to $p \gg L$ (e.g., $p = 100$, $L = 24$) would force algorithms to be mid-convergence at every layer.

### 3.3 Probe-Behavior Mismatch (Exploratory)

We fit linear probes from model activations to algorithm state variables (e.g., CG iterates $\alpha_\ell$, GD weight vectors $\mathbf{w}_\ell$). From the mixed-$\kappa$ trained model:

- **GD probes** achieve higher cosine similarity than CG probes at every $\kappa$ (mean 0.11 vs 0.06)
- Yet the model's **convergence rate** matches CG, not GD

This mismatch---GD-like internal representations but CG-like behavior---is suggestive but must be interpreted cautiously given several limitations of our probe setup. Possible explanations:

1. The model implements CG-like optimization but stores states in a representation basis that happens to align better with GD state variables.
2. The model implements something genuinely different from all six algorithms (not in our comparison set), which happens to have CG-like convergence.
3. Linear probes are insufficient; the CG-like structure may be recoverable with nonlinear probes.

**Important caveat.** These probes are linear ridge regressions. The low absolute cosine similarity values (0.06--0.14) suggest that neither CG nor GD states are strongly encoded in a linearly accessible form. Nonlinear probes, basis controls, and distribution-shift tests are needed before drawing mechanistic conclusions from this mismatch.


## 4. Silent Failure of Softmax Attention as a Kernel Regressor

Independent of the algorithm identification question, we characterize when softmax attention---viewed as an approximate exponential kernel regressor---fails silently.

Softmax attention computes $f(x_q) = \sum_i w_i y_i$ where $w_i = \exp(x_q^\top x_i / \tau) / Z$. This is a Nadaraya-Watson estimator with an exponential kernel. We compare it against the exact exponential-kernel KRR oracle.

**Four failure axes:**

| Regime | RMSE ratio | Rank corr. $\rho$ | Failure mode |
|---|---|---|---|
| Healthy ($\tau=1$, $p=8$) | 0.38x | 0.71 | Adequate |
| High dim ($p=64$) | 1.19x | **0.22** | **Silent**: RMSE plausible, ordering destroyed |
| Low temp ($\tau=0.05$) | 0.89x | 0.72 | Moderate: 1-NN collapse |
| Ill-cond ($\kappa=500$) | 5.6e-13x | **0.24** | **Silent**: catastrophic but RMSE hides it |

The "high dimension" and "ill-conditioned" cases are *silent* failures: the softmax RMSE is comparable to or better than the oracle RMSE, yet rank correlation drops to 0.22--0.24, meaning the prediction *ordering* is nearly random. Standard evaluation by RMSE alone would miss this entirely.

**Mechanism.** In high dimensions, inner products $\langle x_i, x_j \rangle$ concentrate around 0 for random features. Then $\exp(0/\tau) = 1$ for all pairs, and softmax weights become uniform: $w_i \approx 1/n$. The prediction collapses to $\bar{y}$, which has low RMSE if labels have low variance, but zero discriminative power.


## 5. Related Work

**ICL as optimization.** Von Oswald et al. (2023) and Akyurek et al. (2023) established that ICL implements GD on linear tasks. Fu et al. (2023) showed trained models achieve second-order convergence rates. Ahn et al. (2024, NeurIPS) proved transformers learn preconditioned GD with a data-covariance preconditioner. Our work extends this line by testing six algorithms simultaneously and showing the preconditioned-GD identification is not uniquely supported at current scales---CG and preconditioned CG are equally consistent.

**ICL theory.** Mahankali et al. (2023) proved one-layer optimality of GD. Bai et al. (2023) showed transformers can implement algorithm selection across function classes. Park et al. (2024) identified competing algorithmic phases with transitions. Our condition-number-stratified design connects to this "which algorithm when?" question.

**Kernel perspectives.** The connection between attention and kernel regression (relating transformers to ridge regression) provides theoretical grounding for our softmax-as-kernel analysis. Our silent failure characterization identifies where this connection breaks down.


## 6. Discussion

**What we can claim.** Trained ICL transformers implement optimization in the CG convergence class: dramatically faster than GD, matching second-order rates. This is robust across condition numbers, model scales, and evaluation metrics.

**What we cannot claim.** The specific algorithm (CG vs. preconditioned CG vs. preconditioned GD) is not identifiable at experimental scales where $p \leq L$. This suggests the specificity of prior claims, including Ahn et al.'s "preconditioned GD," requires further investigation at larger scales.

**The probe-behavior mismatch.** Our finding that linear probes recover GD-like states more easily than CG-like states, despite CG-like convergence, is a suggestive observation that we present as exploratory. Resolving this requires nonlinear probes, basis controls, and distribution-shift tests that we leave to future work.

**Implications for practice.** The silent failure modes of softmax attention---where RMSE looks acceptable but rank ordering is destroyed---have practical implications for deploying ICL in settings with high-dimensional features or ill-conditioned data. Evaluating ICL systems by RMSE alone is insufficient; rank correlation and tail-error metrics are essential.

**Limitations and future work.** (1) Our experiments use synthetic linear regression; extending to nonlinear tasks and pretrained LLMs is needed. (2) Scaling to $p \gg L$ (e.g., $p = 100$ with 24 layers) may enable algorithm discrimination. (3) Nonlinear probes (MLP) for CG states could resolve the probe-behavior mismatch.


## References

- Ahn, K., et al. (2024). Transformers learn to implement preconditioned gradient descent for in-context learning. *NeurIPS 2024*.
- Akyurek, E., et al. (2023). What learning algorithm is in-context learning? Investigations with linear models. *ICLR 2023*.
- Bai, Y., et al. (2023). Transformers as statisticians: Provable in-context learning with in-context algorithm selection. *NeurIPS 2023*.
- Dai, D., et al. (2023). Why can GPT learn in-context? Language models implicitly perform gradient descent as meta-optimizers. *ACL Findings 2023*.
- Fu, D., et al. (2023). Transformers learn higher-order optimization methods for in-context learning: A study with linear models. *arXiv:2310.17086*.
- Garg, S., et al. (2022). What can transformers learn in-context? A case study of simple function classes. *NeurIPS 2022*.
- Mahankali, A., et al. (2023). One step of gradient descent is provably the optimal in-context learner with one layer. *arXiv:2307.03576*.
- Park, S., et al. (2024). Competition dynamics shape algorithmic phases of in-context learning. *arXiv:2405.16751*.
- von Oswald, J., et al. (2023). Transformers learn in-context by gradient descent. *ICML 2023*.
