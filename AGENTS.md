# Project Conventions — MetaRepICL

## Build & Verification Commands

```bash
# Run all tests (43 tests, ~3s)
python -m pytest tests/ -v

# Quick smoke test (algorithm convergence only)
python -m pytest tests/test_algorithm_id.py -v

# Run algorithm identification on pretrained model
python experiments/algorithm_id.py --load docs/figures/trained_mixed/model_mixed.pt

# Quick test run (5000 steps)
python experiments/algorithm_id.py --steps 5000

# Run discrimination sweep (quick mode for testing)
python experiments/discrimination_sweep.py --quick

# Run multi-seed analysis
python experiments/multi_seed.py --seeds 42 123 456

# Run metric robustness check
python experiments/metric_robustness.py --load docs/figures/trained_mixed/model_mixed.pt

# Run probe sanity suite
python experiments/probe_sanity.py --load docs/figures/trained_mixed/model_mixed.pt

# Run architecture robustness (quick)
python experiments/arch_robustness.py --quick

# Run silent failure analysis (no model needed)
python experiments/silent_failure.py
```

## Code Conventions

- **Seeds**: Always set via `torch.manual_seed(seed)` + `np.random.seed(seed)`. Bootstrap uses `np.random.default_rng(42 + l)`.
- **Data generation**: `generate_batch_with_cov()` creates controlled condition number via log-spaced eigenvalues. Mixed-kappa training draws uniformly from `KAPPAS`.
- **Algorithms**: All operate in feature space: `(X^T X + lam I) w = X^T y`. Defined in `experiments/algorithm_id.py`.
- **Model**: `ICLTransformer` from `src/models/icl_transformer.py` — pre-norm GPT-style, full attention, batch_first.
- **Metrics**: R^2 with 200-iteration bootstrap 95% CI. MSE profile distance as secondary metric.
- **Plotting**: Always `matplotlib.use('Agg')`, save to `out_dir/`, use `ALGO_COLORS` for consistency.
- **Checkpoints**: Saved as `{'model': state_dict, 'cfg': cfg}` via `torch.save()`.
- **Probes**: Linear ridge regression (lambda=1e-3), cosine similarity metric.

## Key Design Decisions

- Feature-space algorithms (not data-space) since transformers see raw X
- Per-layer readout heads trained separately (3000 steps, Adam, lr=1e-3)
- Kappa-weighted mean R^2 emphasizes harder (high-kappa) problems
- Bootstrap CIs on R^2 for formal distinguishability testing

## Workshop Paper Thesis

> We can reliably distinguish first-order from CG-class behavior in transformer ICL
> on linear regression, but cannot uniquely identify the specific member of that class
> under standard observational comparisons at current experimental scale.

## Language Rules

- Do NOT use "CG-class confirmed" — use "CG-class best fits"
- Do NOT use "probe-convergence tension" — use "probe-behavior mismatch"
- Present probe results as "exploratory" until MLP probes + controls are done
- Use "consistent with" not "confirming" for second-order convergence claims
