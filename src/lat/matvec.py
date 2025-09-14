from typing import Tuple
import numpy as np


def attention_matvec(phi_queries: np.ndarray, phi_keys: np.ndarray, values: np.ndarray) -> np.ndarray:
	"""Compute unnormalized dot-product attention mat-vec.
	Given phi_queries (q tokens × p), phi_keys (k tokens × p), and values v (k tokens,)
	returns (K v)_j for each query j, where K = phi_queries @ phi_keys.T and v broadcast.
	If values is (k,d), performs mat-vec per column returning (q,d).
	"""
	assert phi_queries.ndim == 2 and phi_keys.ndim == 2
	assert phi_keys.shape[1] == phi_queries.shape[1]
	q, p = phi_queries.shape
	k, p2 = phi_keys.shape
	assert p2 == p
	Kv = phi_queries @ phi_keys.T
	if values.ndim == 1:
		assert values.shape[0] == k
		return Kv @ values
	else:
		assert values.shape[0] == k
		return Kv @ values
