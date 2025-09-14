from typing import Tuple
import numpy as np


def reductions_sum_squares(vec: np.ndarray) -> float:
	"""Compute sum of squares ∑_i v_i^2 as would an aggregator token, in float64."""
	v = np.asarray(vec, dtype=np.float64)
	return float(np.dot(v, v))


def reductions_dot(vec1: np.ndarray, vec2: np.ndarray) -> float:
	"""Compute dot product ∑_i v1_i * v2_i in float64."""
	v1 = np.asarray(vec1, dtype=np.float64)
	v2 = np.asarray(vec2, dtype=np.float64)
	assert v1.shape == v2.shape
	return float(np.dot(v1, v2))


def broadcast_scalar_to_tokens(scalar: float, num_tokens: int) -> np.ndarray:
	"""Broadcast a scalar back to tokens (e.g., via aggregator token write-back)."""
	return np.full(shape=(num_tokens,), fill_value=float(scalar), dtype=np.float64)
