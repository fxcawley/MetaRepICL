import numpy as np
from src.lat.cg_step import cg_step


def reference_cg_step(alpha, r, p, K, lam):
	Ap = K @ p + lam * p
	rr = float(r @ r)
	pAp = float(p @ Ap)
	gamma = rr / pAp
	alpha_next = alpha + gamma * p
	r_next = r - gamma * Ap
	rr_next = float(r_next @ r_next)
	beta = rr_next / rr
	p_next = r_next + beta * p
	return alpha_next, r_next, p_next


def test_cg_step_matches_reference():
	rng = np.random.default_rng(2024)
	n = 12
	K = rng.standard_normal((n, n)).astype(np.float64)
	K = 0.5 * (K + K.T)  # symmetrize
	lam = 1e-1
	y = rng.standard_normal(n).astype(np.float64)
	alpha = np.zeros(n, dtype=np.float64)
	r = y.copy()  # assuming initial alpha=0 => r = y - A alpha = y
	p = r.copy()
	Ap = K @ p + lam * p
	alpha_ref, r_ref, p_ref = reference_cg_step(alpha, r, p, K, lam)
	alpha_out, r_out, p_out, rr_next = cg_step(alpha, r, p, Ap, lam)
	assert np.allclose(alpha_out, alpha_ref, rtol=1e-12, atol=1e-12)
	assert np.allclose(r_out, r_ref, rtol=1e-12, atol=1e-12)
	assert np.allclose(p_out, p_ref, rtol=1e-12, atol=1e-12)
