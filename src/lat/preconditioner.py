import numpy as np
from typing import Tuple


def build_diag_preconditioner(diag: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""Given an estimate of the diagonal of A, return P^{-1} diagonal entries."""
	d = np.asarray(diag, dtype=np.float64)
	return 1.0 / (d + eps)


def apply_preconditioner(vec: np.ndarray, pinv_diag: np.ndarray) -> np.ndarray:
	"""Apply P^{-1} (diagonal) to a vector."""
	v = np.asarray(vec, dtype=np.float64)
	p = np.asarray(pinv_diag, dtype=np.float64)
	assert v.shape == p.shape
	return p * v
