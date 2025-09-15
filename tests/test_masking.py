import numpy as np
from src.lat.masking import build_roles, attention_mask


def test_compute_phase_blocks_query_to_support():
	k = 5
	roles = build_roles(k)
	M = attention_mask(roles, phase='compute')
	# query index is k
	q = k
	# support indices 0..k-1 should be blocked as targets from query
	for j in range(k):
		assert M[q, j] < -1e8


def test_readout_phase_allows_query_to_support():
	k = 5
	roles = build_roles(k)
	M = attention_mask(roles, phase='readout')
	q = k
	for j in range(k):
		assert M[q, j] == 0.0
