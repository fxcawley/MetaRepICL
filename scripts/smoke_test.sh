#!/usr/bin/env bash
# Smoke test: verify core imports, run minimal computations, exit non-zero on failure
set -euo pipefail

echo "=== MetaRep Smoke Test ==="

echo "[1/5] Checking Python imports..."
python -c "
import torch, numpy, scipy
from src.lat.matvec import attention_matvec
from src.lat.cg_stack import run_cg
from src.data.synth_linear import make_linear_dataset, krr_oracle
from src.eval.metrics import rmse, mean_ci
print('  Imports OK')
"

echo "[2/5] Running data generator..."
python -c "
from src.data.synth_linear import make_linear_dataset
Xs, ys, Xq, yq = make_linear_dataset(n_support=8, n_query=4, p=4, noise=0.1, seed=42)
assert Xs.shape == (8, 4), f'Bad shape: {Xs.shape}'
print('  Data generator OK')
"

echo "[3/5] Running CG stack..."
python -c "
import numpy as np
from src.lat.cg_stack import run_cg
phi = np.random.default_rng(42).standard_normal((8, 4))
y = np.random.default_rng(42).standard_normal(8)
alpha, r, p = run_cg(phi, y, lam=0.1, t=3)
assert alpha.shape == (8,), f'Bad alpha shape: {alpha.shape}'
print('  CG stack OK')
"

echo "[4/5] Running metrics..."
python -c "
from src.eval.metrics import rmse, mean_ci
import numpy as np
err = rmse(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
assert err < 0.2, f'Bad RMSE: {err}'
ci = mean_ci([0.1, 0.12, 0.11])
assert 'mean' in ci and 'ci' in ci, f'Bad CI: {ci}'
print('  Metrics OK')
"

echo "[5/5] Running unit tests (fast subset)..."
python -m pytest tests/test_lat_matvec.py tests/test_synth_linear.py -q --tb=short

echo "=== Smoke test PASSED ==="
