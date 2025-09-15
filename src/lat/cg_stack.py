import numpy as np
from typing import Tuple
from .matvec import attention_matvec
from .cg_step import cg_step


def run_cg(
	phi: np.ndarray,
	y: np.ndarray,
	lam: float,
	t: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Run t CG steps solving (K + lam I) alpha = y, where K = ΦΦ^T.
	Returns (alpha_t, r_t, p_t).
	"""
	phi = np.asarray(phi, dtype=np.float64)
	y = np.asarray(y, dtype=np.float64)
	n = phi.shape[0]
	alpha = np.zeros(n, dtype=np.float64)
	r = y.copy()
	p = r.copy()
	for _ in range(int(t)):
		Kp = attention_matvec(phi, phi, p)
		Ap = Kp + lam * p
		alpha, r, p, _ = cg_step(alpha, r, p, Ap, lam)
	return alpha, r, p
