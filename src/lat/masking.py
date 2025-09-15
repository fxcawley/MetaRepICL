import numpy as np
from typing import Literal


Role = Literal['support', 'query', 'aggregator']


def build_roles(num_support: int) -> np.ndarray:
	"""Return an array of roles for tokens: [support * k, query, aggregator]."""
	roles = np.array(['support'] * num_support + ['query', 'aggregator'], dtype=object)
	return roles


def attention_mask(roles: np.ndarray, phase: Literal['compute', 'readout']) -> np.ndarray:
	"""Construct an attention mask where mask[i,j]=0 allows i to attend to j; -inf blocks.
	- During 'compute' phase: supports can attend supports and aggregator; aggregator can attend supports; query cannot attend supports or aggregator.
	- During 'readout' phase: query can attend supports and aggregator; supports cannot attend query.
	"""
	n = roles.shape[0]
	M = np.zeros((n, n), dtype=np.float64)
	neg_inf = -1e9
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			ri = roles[i]
			rj = roles[j]
			if phase == 'compute':
				if ri == 'support' and rj in ('support', 'aggregator'):
					pass
				elif ri == 'aggregator' and rj == 'support':
					pass
				else:
					M[i, j] = neg_inf
			elif phase == 'readout':
				if ri == 'query' and rj in ('support', 'aggregator'):
					pass
				elif ri == 'support' and rj == 'query':
					M[i, j] = neg_inf
				else:
					# allow supports↔supports and aggregator↔supports if needed
					pass
	return M
