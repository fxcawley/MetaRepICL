from experiments.precond import run_precond, run_preconditioned_cg
import numpy as np


def test_preconditioner_not_worse_monotone():
	res = run_precond(t=5)
	eu = res["err_un"]
	ep = res["err_pr"]
	assert len(eu) == len(ep) == 5
	# Weak check: final error with preconditioner is not worse than unpreconditioned
	assert ep[-1] <= eu[-1] + 1e-8


def test_preconditioned_cg_converges():
	"""Verify that preconditioned CG actually converges to the correct solution."""
	rng = np.random.default_rng(42)
	# Use p >= n so K is full rank (well-conditioned system)
	n, p = 16, 32
	phi = rng.standard_normal((n, p)).astype(np.float64)
	K = phi @ phi.T
	lam = 0.1
	y = rng.standard_normal(n).astype(np.float64)
	alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
	# Jacobi preconditioner
	M_inv_diag = 1.0 / (np.diag(K) + lam + 1e-6)
	alpha_pr, hist = run_preconditioned_cg(K, y, lam, t=20, M_inv_diag=M_inv_diag)
	err = np.linalg.norm(alpha_pr - alpha_star)
	assert err < 1e-6, f"Preconditioned CG did not converge: err={err}"
