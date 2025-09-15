import numpy as np
from src.lat.cg_stack import run_cg


def test_cg_error_decreases_with_t():
	rng = np.random.default_rng(0)
	n, p = 20, 6
	phi = rng.standard_normal((n, p)).astype(np.float64)
	K = phi @ phi.T
	lam = 1e-1
	y = rng.standard_normal(n).astype(np.float64)
	# Oracle solution
	alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
	def err(t):
		alpha_t, _, _ = run_cg(phi, y, lam, t)
		return float(np.linalg.norm(alpha_t - alpha_star))
	e1 = err(1)
	e3 = err(3)
	e6 = err(6)
	assert e3 <= e1 + 1e-9
	assert e6 <= e3 + 1e-9
