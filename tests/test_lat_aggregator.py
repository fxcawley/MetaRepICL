import numpy as np
from src.lat.aggregator import reductions_sum_squares, reductions_dot, broadcast_scalar_to_tokens


def test_reductions_sum_squares_and_dot():
	rng = np.random.default_rng(7)
	r = rng.standard_normal(13).astype(np.float64)
	p = rng.standard_normal(13).astype(np.float64)
	Ap = rng.standard_normal(13).astype(np.float64)
	s1 = reductions_sum_squares(r)
	assert np.allclose(s1, np.sum(r * r), rtol=1e-12, atol=1e-12)
	s2 = reductions_dot(p, Ap)
	assert np.allclose(s2, float(np.sum(p * Ap)), rtol=1e-12, atol=1e-12)


def test_broadcast_scalar():
	s = 3.14159
	out = broadcast_scalar_to_tokens(s, num_tokens=5)
	assert out.shape == (5,)
	assert np.allclose(out, np.array([s] * 5, dtype=np.float64), rtol=0, atol=0)
