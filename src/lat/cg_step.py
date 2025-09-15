from typing import Tuple
import numpy as np


def cg_step(
	alpha: np.ndarray,
	r: np.ndarray,
	p: np.ndarray,
	Ap: np.ndarray,
	lam: float,
	prev_rr: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
	"""Perform one CG step for (K + lam I) alpha = y.
	Inputs are float64 vectors per support token. Ap should be (K p) + lam p.
	If prev_rr is None, compute from r.
	Returns (alpha_next, r_next, p_next, rr_next).
	"""
	alpha = np.asarray(alpha, dtype=np.float64)
	r = np.asarray(r, dtype=np.float64)
	p = np.asarray(p, dtype=np.float64)
	Ap = np.asarray(Ap, dtype=np.float64)

	rr = float(r @ r) if prev_rr is None else float(prev_rr)
	pAp = float(p @ Ap)
	# Guard against division by zero
	gamma = rr / (pAp + 1e-18)
	alpha_next = alpha + gamma * p
	r_next = r - gamma * Ap
	rr_next = float(r_next @ r_next)
	beta = rr_next / (rr + 1e-18)
	p_next = r_next + beta * p
	return alpha_next, r_next, p_next, rr_next
