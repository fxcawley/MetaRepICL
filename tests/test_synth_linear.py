import numpy as np
from src.data.synth_linear import make_linear_dataset, krr_oracle


def test_krr_shapes_and_basic():
	Xs, ys, Xq, yq = make_linear_dataset(n_support=16, n_query=8, p=5, noise=0.0, seed=123)
	pred = krr_oracle(Xs, ys, Xq, lam=1e-3)
	assert pred.shape == (8,)
	# With zero noise and small lambda, correlation should be high
	corr = np.corrcoef(pred, yq)[0, 1]
	assert corr > 0.9
