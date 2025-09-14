import numpy as np
from src.lat.matvec import attention_matvec


def test_attention_matvec_matches_reference():
	# float64 to enforce strict equality tolerances
	rng = np.random.default_rng(123)
	q, k, p = 7, 11, 5
	phi_q = rng.standard_normal((q, p)).astype(np.float64)
	phi_k = rng.standard_normal((k, p)).astype(np.float64)
	v = rng.standard_normal((k,)).astype(np.float64)
	K = phi_q @ phi_k.T
	ref = K @ v
	out = attention_matvec(phi_q, phi_k, v)
	assert out.shape == (q,)
	assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_attention_matvec_matrix_values():
	rng = np.random.default_rng(456)
	q, k, p, d = 4, 6, 3, 2
	phi_q = rng.standard_normal((q, p)).astype(np.float64)
	phi_k = rng.standard_normal((k, p)).astype(np.float64)
	V = rng.standard_normal((k, d)).astype(np.float64)
	ref = (phi_q @ phi_k.T) @ V
	out = attention_matvec(phi_q, phi_k, V)
	assert out.shape == (q, d)
	assert np.allclose(out, ref, rtol=1e-12, atol=1e-12)
