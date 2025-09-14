import numpy as np
from typing import Tuple


def make_linear_dataset(n_support: int, n_query: int, p: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Generate deterministic linear data with controllable dimensions.
	Returns (X_support, y_support, X_query, y_query).
	"""
	rng = np.random.default_rng(seed)
	X = rng.standard_normal(size=(n_support + n_query, p)).astype(np.float64)
	w = rng.standard_normal(size=(p,)).astype(np.float64)
	y = X @ w + noise * rng.standard_normal(size=(n_support + n_query))
	Xs, Xq = X[:n_support], X[n_support:]
	ys, yq = y[:n_support], y[n_support:]
	return Xs, ys, Xq, yq


def krr_oracle(Xs: np.ndarray, ys: np.ndarray, Xq: np.ndarray, lam: float) -> np.ndarray:
	"""Closed-form KRR predictions using a dot-product kernel on raw features.
	This is for quick sanity checks; in the project we use φ from lower layers.
	"""
	K = Xs @ Xs.T
	A = K + lam * np.eye(K.shape[0], dtype=np.float64)
	alpha = np.linalg.solve(A, ys)
	kq = Xs @ Xq.T
	return (kq.T @ alpha)
