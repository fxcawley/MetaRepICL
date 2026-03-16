# Workshop Submission Action Plan

## Thesis (reviewer-proof version)

> We can reliably distinguish first-order from CG-class behavior in transformer ICL
> on linear regression, but cannot uniquely identify the specific member of that class
> under standard observational comparisons at current experimental scale.

---

## Priority Tiers

### Tier 0: Must-do before submission (5 items)

| # | Experiment | Script | What it shows | Est. compute |
|---|-----------|--------|--------------|-------------|
| 1 | **p >> L discrimination sweep** | `experiments/discrimination_sweep.py` | Either CG/PCG/PGD separate at large p, or the identifiability limit is structural | 2-8 GPU-hours |
| 2 | **Multi-metric robustness** | `experiments/metric_robustness.py` | Conclusions stable across MSE, RMSE, rank correlation, pairwise wins, parameter error | Minutes (analysis only) |
| 3 | **Multi-seed robustness** | `experiments/multi_seed.py` | "Model >> GD" and "top-3 tie" are not a seed=42 story | 3-5x base compute |
| 4 | **Probe section rework** | `experiments/probe_sanity.py` | Either strengthen probes with MLP + controls, or demote to exploratory | 1-2 GPU-hours |
| 5 | **Paper language + title** | Manual edits | Remove "CG-class confirmed", fix KRR framing, soften probe claims | 30 min |

### Tier 1: Strongly recommended

| # | Experiment | Script | What it shows |
|---|-----------|--------|--------------|
| 6 | **Architecture robustness** | `experiments/arch_robustness.py` | Findings not specific to one (depth, width, heads) config |
| 7 | **Data-distribution robustness** | Extend `discrimination_sweep.py` | Findings not specific to geometric eigenvalue spacing |
| 8 | **Adversarial separation instances** | Section in `discrimination_sweep.py` | Problem families where CG vs PGD *should* diverge |

### Tier 2: Nice-to-have

| # | Experiment | What it shows |
|---|-----------|--------------|
| 9 | Training horizon variation | Algorithm ID stable across 10K-100K steps |
| 10 | Optimizer variation | AdamW vs SGD+momentum training doesn't change conclusions |

---

## Experiment Specifications

### 1. Discrimination Sweep (`experiments/discrimination_sweep.py`)

**Goal**: Test whether p >> L enables algorithm discrimination.

**Sweep grid**:
```
p:          [10, 20, 40, 60, 80, 100]
num_layers: [12, 24]
n_support:  2 * p  (ensures n > p for each)
kappas:     [10, 50, 100, 500, 1000]
n_per_kappa: 300
seeds:      [42, 123, 456]
```

**Key design**: At p=100, L=24, the model is mid-convergence at every layer.
CG converges in <= p steps; GD barely moves. If CG/PCG/PGD still cluster,
it's structural. If one wins, that's the headline result.

**Additional data-distribution conditions**:
- Isotropic Gaussian (kappa=1 control)
- Geometric spectrum (current default)
- Step spectrum (eigenvalues = {1, kappa}, half each)
- Mixture: 50% kappa=10, 50% kappa=500 within-batch

**Metrics**: All from metric_robustness.py applied at each grid point.

**Output**: Heatmap of "best algorithm" and "gap vs second" across (p, L, kappa).

**Predicted outcomes**:
- Outcome A: At p=100, L=24, one algorithm (likely PCG or CG) wins decisively.
  Paper claim: "Algorithm identification becomes possible when p >> L."
- Outcome B: Top 3 remain clustered even at p=100.
  Paper claim: "Indistinguishability is structural, not just underpowered."
- Either is publishable. Not running it is not.

### 2. Metric Robustness (`experiments/metric_robustness.py`)

**Goal**: Show conclusions are stable across metrics.

**Metrics computed** (given existing per-problem prediction data):
1. Per-problem R^2 (existing)
2. MSE profile distance (existing)
3. **NEW: RMSE on predictions** (model vs algo predictions)
4. **NEW: Spearman rank correlation** of per-problem errors
5. **NEW: Per-layer improvement correlation** (does the model improve on the same problems?)
6. **NEW: Pairwise wins** (for each problem, which algo is closer to model?)
7. **NEW: Final-iterate parameter error** ||w_model - w_algo|| where w_model from readout

**Output**: Table showing all metrics agree on: (a) GD clearly worse, (b) top 3 clustered.

### 3. Multi-Seed (`experiments/multi_seed.py`)

**Goal**: Confirm findings across seeds.

**Design**: Run `algorithm_id.py` with seeds [42, 123, 456, 789, 2024].
For each seed: full training + algorithm ID. Report:
- Mean + std of weighted R^2 per algorithm across seeds
- "Model >> GD" gap with CI across seeds
- "Top 3 gap" with CI across seeds
- Any seed where ranking changes

### 4. Probe Sanity Suite (`experiments/probe_sanity.py`)

**Goal**: Either strengthen probe claims or provide evidence to demote them.

**Components**:
1. **Capacity sweep**: Linear probe vs 1-hidden-layer MLP (64, 256 units) vs 2-layer MLP
   - If MLP recovers CG states that linear probe misses: "nonlinear encoding"
   - If MLP also fails: "CG states genuinely not present"
2. **Layer-wise CIs**: Bootstrap CI on probe cosine similarity per layer
3. **Control targets**: 
   - Random targets (existing)
   - Ridge solution w* (should be recoverable at late layers)
   - Raw y_support (trivially available, sanity check)
4. **Basis control**: Rotate CG states by random orthogonal Q. Linear probe on
   Q @ z_cg should get same cosine sim (if probe is finding the right subspace)
5. **Distribution shift**: Train probe on kappa=10, test on kappa=100

### 5. Paper Language Edits

**README.md**:
- Change title: "MetaRep -- Algorithm Identification in In-Context Learning" (drop KRR)
- "CG-class confirmed" -> "CG-class best fits among tested algorithms"
- "Probe-convergence tension" -> "Probe-behavior mismatch (exploratory)"

**paper/workshop_paper.md**:
- Title: "Limits of Algorithm Identification in In-Context Linear Regression"
- Abstract: Reframe around identification limits, not confirmation
- Section 3.1: "CG-class confirmed" -> "CG-class best fits"
- Section 3.3: Add caveat about linear-only probes, frame as exploratory
- Discussion: Soften mechanistic tension to "suggestive mismatch"

### 6. Architecture Robustness (`experiments/arch_robustness.py`)

**Sweep** (3-4 configs, not a giant grid):
```
Base:    12 layers, 256 dim, 4 heads, p=10
Deep:    24 layers, 256 dim, 4 heads, p=10
Wide:    12 layers, 512 dim, 4 heads, p=10
Heads:   12 layers, 256 dim, 8 heads, p=10
```
Train each, run algorithm_id. Show rankings are stable.

---

## Suggested Figures for Workshop Paper

1. **Figure 1**: Algorithm ID summary bar chart (existing, improved with error bars across seeds)
2. **Figure 2**: Discrimination sweep heatmap — "best algorithm gap" as function of (p, L)
3. **Figure 3**: Multi-metric agreement table (all metrics show same ranking)
4. **Figure 4**: Probe results with MLP comparison + CIs (demoted to appendix if weak)

---

## Concrete Checklist

- [ ] Run discrimination sweep (p >> L)
- [ ] Add 5+ alternative metrics, show conclusions stable
- [ ] Add 3-5 seed robustness for main ranking
- [ ] Rework probes with MLP + controls, or demote to exploratory
- [ ] Rewrite title/abstract: paper is about identification limits, not KRR
- [ ] Architecture robustness (3-4 configs)
- [ ] Data-distribution robustness (at least 2 spectrum types)
- [ ] Run all tests, ensure nothing breaks
- [ ] Generate final figures
- [ ] LaTeX build (or clean markdown submission)

---

## What to Cut/Soften

| Current | Replace with |
|---------|-------------|
| "CG-class confirmed" | "CG-class best fits among tested algorithms" |
| "probe-convergence tension" | "probe-behavior mismatch" |
| "challenges specificity of prior claims" | "suggests identification may require larger scale" |
| Any KRR framing in title/header | Algorithm identification framing |
| "mechanistic tension unreported in prior work" | "suggestive mismatch between behavior and linear probes" |

## What I Believe (honest assessment)

**Strong results**:
- "Model >> GD" — large, clear, will survive review
- "Top 3 indistinguishable at current scale" — honest and valuable

**Needs more work**:
- Probe-convergence tension — only trustworthy with MLP + controls
- Any exact mechanistic claim — premature

**Best paper structure**: One clean claim about identification limits,
with the probe result as an exploratory finding (Section 5 or appendix).
