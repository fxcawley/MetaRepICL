import json
import argparse
import numpy as np
from typing import List, Dict
import hydra
from omegaconf import DictConfig

# Patch argparse for Hydra/Py3.14
_orig_add_argument = argparse.ArgumentParser.add_argument
def _safe_add_argument(self, *args, **kwargs):
	if 'help' in kwargs:
		h = kwargs['help']
		if hasattr(h, '__class__') and h.__class__.__name__ == 'LazyCompletionHelp':
			kwargs['help'] = "Shell completion"
	return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _safe_add_argument


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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	parser = argparse.ArgumentParser()
	parser.add_argument("--plot", action="store_true")
	parser.add_argument("--out", type=str, default="docs/figures/sweeps/width_rank_curve.png")
	
	# Handle arguments: if --plot passed in sys.argv, we want to respect it.
	
	# Since we are using Hydra, let's prefer cfg values.
	plot = bool(cfg.get("plot", False))
	out = str(cfg.get("out", "docs/figures/sweeps/width_rank_curve.png"))
	
	# Also check argparse for direct invocation compatibility if users use flags
	try:
		args, _ = parser.parse_known_args()
		if args.plot: plot = True
		if args.out != "docs/figures/sweeps/width_rank_curve.png": out = args.out
	except Exception:
		pass

	res = run_width_rank(
		seed=int(cfg.get("seed", 123)),
		n=int(cfg.get("n_support", 64)),
		p=int(cfg.get("p", 32)),
		lam=float(cfg.get("lambda", 1e-2)),
		noise=float(cfg.get("noise", 0.1)),
	)
	print(json.dumps(res))
	
	if plot:
		try:
			import matplotlib.pyplot as plt
			import os
			ms = res["m"]
			errs = res["pred_err"]
			plt.figure()
			plt.plot(ms, errs, marker='o')
			plt.xlabel('width m (rank of sketch)')
			plt.ylabel('prediction error vs oracle')
			plt.title('Width–rank empirical')
			plt.tight_layout()
			
			os.makedirs(os.path.dirname(out), exist_ok=True)
			plt.savefig(out, dpi=150)
			print(f"Saved plot to {out}")
		except Exception as e:
			print(f"Error plotting: {e}")


if __name__ == "__main__":
	main()
