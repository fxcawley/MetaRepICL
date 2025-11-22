<<<<<<< HEAD
## Reproducibility Guide (MetaRep)

### Environment
- OS: Linux/Windows
- Python: 3.10+
- CUDA: 12.x (if GPU)
- Manager: conda or venv
- Container: Docker optional (CUDA base image); see Dockerfile and lockfile for exact versions.

#### Setup
```bash
conda create -n metarep python=3.10 -y
conda activate metarep
pip install -r requirements.txt
```

### One-Step Reproduction
To reproduce the key figures in the paper (Figure 1: Route A MVP, Figure 2: Failure Modes, Figure 3: Ablations), run the provided bash script:

```bash
# Must be run from the repository root
bash scripts/reproduce_figure_1.sh
```

This will generate:
- `figures/route_a_mvp.png`
- `figures/failure_modes/ill_conditioned_cg.png`
- `figures/ablations/route_b_heads.png`
- `figures/ablations/route_a_temp.png`

### Determinism and Seeds
- Default seeds: 123, 456, 789 (main results use 3 seeds; ablations use 5 distinct seeds).
- Enable cudnn deterministic and disable benchmarking.
- Log Python, NumPy, PyTorch, CUDA versions and git commit.
- Enforce seed presence and config hash via pre-commit hook; runs without explicit seed are blocked.

### Data
- Synthetic generators: deterministic given seed; no external downloads required.
- Language-style numeric tasks: small, permissive datasets (TBD). Scripts will auto-download to `data/`.

### Running Experiments
Minimal examples (placeholders; actual paths provided in repo):
```bash
python experiments/route_a_minimal.py +seed=123 trainer.max_steps=2000
python experiments/width_rank.py +seed=123 model.width=64,32,16
python experiments/probes/state_probes.py +seed=123
```

### Checkpoints and Resumption
- Checkpoint every 500–1000 steps.
- Resumption validated to within 1% metric drift.

### Evaluation
```bash
python experiments/run_eval.py +seeds="[123,456,789]" eval.report_ci=true
```
Reports include mean ± 95% CI and paired tests where applicable.

### Artifact Release
- Code: MIT License
- Models/figures: CC BY 4.0
- Include `scripts/reproduce_figure_1.sh` for end-to-end reproduction of the LAT→CG demo figure.
- Double-blind policy: For AISTATS, do not cite or link an arXiv preprint in the submission; defer public, author-attributed repo releases until after decisions or provide an anonymized artifact.

### Reporting
- Report mean ± 95% CI across seeds.
- Include paired tests for A vs B comparisons when runs share seeds.

### Numerics policy
- Default runtime precision: float32. Unit tests in float64 with 1e-6 tolerances on toy problems. Parity test requires ≤1% deviation in final predictions (float32 vs float64) on fixed seeds.
=======
## Reproducibility Guide (MetaRep)

### Environment
- OS: Linux/Windows
- Python: 3.10+
- CUDA: 12.x (if GPU)
- Manager: conda or venv
 - Container: Docker optional (CUDA base image); see Dockerfile and lockfile for exact versions.

#### Setup
```bash
conda create -n metarep python=3.10 -y
conda activate metarep
pip install -r requirements.txt
```

### Determinism and Seeds
- Default seeds: 123, 456, 789 (main results use 3 seeds; ablations use 5 distinct seeds).
- Enable cudnn deterministic and disable benchmarking.
- Log Python, NumPy, PyTorch, CUDA versions and git commit.
 - Enforce seed presence and config hash via pre-commit hook; runs without explicit seed are blocked.

### Data
- Synthetic generators: deterministic given seed; no external downloads required.
- Language-style numeric tasks: small, permissive datasets (TBD). Scripts will auto-download to `data/`.

### Running Experiments
Minimal examples (placeholders; actual paths provided in repo):
```bash
python experiments/route_a_minimal.py +seed=123 trainer.max_steps=2000
python experiments/width_rank.py +seed=123 model.width=64,32,16
python experiments/probes/state_probes.py +seed=123
```

### Checkpoints and Resumption
- Checkpoint every 500–1000 steps.
- Resumption validated to within 1% metric drift.

### Evaluation
```bash
python experiments/run_eval.py +seeds="[123,456,789]" eval.report_ci=true
```
Reports include mean ± 95% CI and paired tests where applicable.

### Artifact Release
- Code: MIT License
- Models/figures: CC BY 4.0
- Include `scripts/reproduce_figure_1.sh` for end-to-end reproduction of the LAT→CG demo figure.
 - Double-blind policy: For AISTATS, do not cite or link an arXiv preprint in the submission; defer public, author-attributed repo releases until after decisions or provide an anonymized artifact.

### Reporting
- Report mean ± 95% CI across seeds.
- Include paired tests for A vs B comparisons when runs share seeds.

### Numerics policy
- Default runtime precision: float32. Unit tests in float64 with 1e-6 tolerances on toy problems. Parity test requires ≤1% deviation in final predictions (float32 vs float64) on fixed seeds.


>>>>>>> 2dee55f (feat: Documentation Website (GitHub Pages) #71)
