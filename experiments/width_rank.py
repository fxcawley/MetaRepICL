import json
import numpy as np
from typing import List, Dict


def effective_dimension(K: np.ndarray, lam: float) -> float:
	w, _ = np.linalg.eigh(K)
	w = np.maximum(w, 0.0)
	return float(np.sum(w / (w + lam)))


def sketch_kernel(phi: np.ndarray, m: int, seed: int) -> np.ndarray:
	"""Project φ to m dims via a random Gaussian map (oblivious sketch)."""
	rng = np.random.default_rng(seed)
	p = phi.shape[1]
	U = rng.standard_normal((p, m)) / np.sqrt(m)
	phim = phi @ U
	return phim @ phim.T


def run_width_rank(
	seed: int = 123,
	n: int = 64,
	p: int = 32,
	lam: float = 1e-2,
	noise: float = 0.1,
	ms: List[int] = [8, 16, 24, 32],
) -> Dict[str, List[float]]:
	rng = np.random.default_rng(seed)
	phi = rng.standard_normal((n, p)).astype(np.float64)
	K = phi @ phi.T
	y = rng.standard_normal(n).astype(np.float64) + noise * rng.standard_normal(n).astype(np.float64)
	# Oracle
	alpha = np.linalg.solve(K + lam * np.eye(n), y)
	def pred_err(Khat: np.ndarray) -> float:
		alphahat = np.linalg.solve(Khat + lam * np.eye(n), y)
		return float(np.linalg.norm(alpha - alphahat))
	errs = []
	deffs = []
	for m in ms:
		Khat = sketch_kernel(phi, m=m, seed=seed + m)
		errs.append(pred_err(Khat))
		deffs.append(effective_dimension(K, lam))
	return {"m": ms, "pred_err": errs, "d_eff": deffs}


def main():
	res = run_width_rank()
	print(json.dumps(res))


if __name__ == "__main__":
	main()
