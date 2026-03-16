# MetaRep -- Algorithm Identification in In-Context Learning

How precisely can we identify the optimization algorithm implemented by in-context learning? We train transformers on ICL regression tasks and systematically compare their behavior against six named iterative algorithms with bootstrap confidence intervals.

**Main findings** (see `paper/workshop_paper.md`):

1. **Model >> GD**: Trained transformers converge dramatically faster than gradient descent (R^2 gap = 0.20), confirming second-order optimization.
2. **CG-class confirmed**: Per-problem predictions correlate strongly (R^2 = 0.92) with conjugate gradients, preconditioned CG, and preconditioned GD.
3. **Specific algorithm not identifiable**: CG, Precond CG, and Precond GD are statistically indistinguishable (gap 0.004, CI 0.017). This challenges the specificity of prior claims.
4. **Probe-convergence tension**: Internal representations are more GD-like than CG-like despite CG-like convergence -- a disconnect unreported in prior work.
5. **Silent failure modes**: Softmax attention as a kernel regressor fails silently in high-dimensional and ill-conditioned regimes (RMSE looks fine, rank ordering destroyed).

## Quickstart

```bash
pip install -r requirements.txt

# Run algorithm identification on pretrained model:
python experiments/algorithm_id.py --load docs/figures/trained_mixed/model_mixed.pt

# Train a new model and run analysis:
python experiments/algorithm_id.py --steps 30000 --p 20 --n-support 40 --num-layers 24

# Run silent failure analysis:
python experiments/silent_failure.py

# Run all tests:
python -m pytest tests/ -v
```

## Repo layout

- `paper/workshop_paper.md` the main paper (workshop format)
- `experiments/algorithm_id.py` algorithm identification experiment (6 algorithms, CIs, stratified)
- `experiments/train_and_probe.py` train ICL transformer, probe for CG/GD states
- `experiments/train_mixed_kappa.py` mixed-kappa training with feature-space CG
- `experiments/silent_failure.py` softmax-as-kernel failure mode analysis
- `src/models/` ICL transformer architecture
- `src/lat/` CG implementation (steps, stacking, preconditioning)
- `src/data/` synthetic and GLM data generators
- `tests/` 43 tests covering algorithms, convergence, stability
- `docs/figures/` experimental results and plots
- `configs/` Hydra configs for experiment sweeps

## Key results

### Algorithm Identification (p=20, 24-layer model, 300 problems/kappa)

| Algorithm | Weighted R^2 | 95% CI |
|---|---|---|
| **Precond CG** | **0.922** | 0.008 |
| **CG** | **0.918** | 0.009 |
| **Precond GD** | **0.910** | 0.011 |
| Chebyshev | 0.780 | 0.042 |
| GD | 0.721 | 0.060 |
| Heavy Ball | 0.581 | 0.046 |

Top 3 are indistinguishable (gap 0.004, combined CI 0.017). Model >> GD is clear.

### Silent Failure (softmax attention as kernel regressor)

| Regime | Softmax RMSE | Rank Correlation | Failure |
|---|---|---|---|
| Healthy | 2.5 | 0.71 | None |
| High-dim (p=64) | 8.2 | **0.22** | Silent |
| Ill-conditioned (kappa=500) | 2.8 | **0.24** | Silent |

## Citation

If you use this work, please cite the workshop paper (forthcoming).

## License

MIT
