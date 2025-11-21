import os
import json
import argparse
import numpy as np
from typing import List, Dict
# Patch argparse to handle Hydra's LazyCompletionHelp on Python 3.14
# This must be done before importing hydra or running @hydra.main
_orig_add_argument = argparse.ArgumentParser.add_argument
def _safe_add_argument(self, *args, **kwargs):
	if 'help' in kwargs:
		h = kwargs['help']
		if hasattr(h, '__class__') and h.__class__.__name__ == 'LazyCompletionHelp':
			kwargs['help'] = "Shell completion"
	return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _safe_add_argument

import hydra
from omegaconf import DictConfig

# Ensure we can import from the package
try:
	from src.lat.cg_stack import run_cg
	from experiments.width_rank import sketch_kernel
except ImportError:
	# If running as script without package installation/context
	import sys
	from pathlib import Path
	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	from src.lat.cg_stack import run_cg
	from experiments.width_rank import sketch_kernel


def make_phi(n: int, p: int, seed: int, cond: float) -> np.ndarray:
	rng = np.random.default_rng(seed)
	U, _ = np.linalg.qr(rng.standard_normal((p, p)))
	s = np.geomspace(cond, 1.0, p)
	phi_base = rng.standard_normal((n, p))
	phi = phi_base @ (U * s)
	return phi


def run_sweep(
	seeds: List[int], 
	n: int, 
	p: int, 
	lam_list: List[float], 
	t_list: List[int], 
	cond_list: List[float], 
	m_list: List[int]
) -> Dict:
	out = {}
	# Make keys strings for JSON compatibility
	out_json = {}
	
	for cond in cond_list:
		for lam in lam_list:
			errs_vs_t = []
			for t in t_list:
				errs = []
				for s in seeds:
					phi = make_phi(n, p, s, cond)
					K = phi @ phi.T
					y = np.random.default_rng(s + 1).standard_normal(n).astype(np.float64)
					alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
					alpha_t, _, _ = run_cg(phi, y, lam, t)
					errs.append(float(np.linalg.norm(alpha_t - alpha_star)))
				errs_vs_t.append((t, float(np.mean(errs))))
			out[(cond, lam)] = errs_vs_t
			out_json[f"cond{cond}_lam{lam}"] = errs_vs_t

	# Optional width sweep at fixed cond, lam, t
	width_curves = {}
	if m_list:
		cond = cond_list[0]
		lam = lam_list[0]
		t = t_list[-1]
		for m in m_list:
			errs = []
			for s in seeds:
				phi = make_phi(n, p, s, cond)
				K = phi @ phi.T
				Khat = sketch_kernel(phi, m=m, seed=s + m)
				y = np.random.default_rng(s + 1).standard_normal(n).astype(np.float64)
				alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
				alphahat = np.linalg.solve(Khat + lam * np.eye(n), y)
				errs.append(float(np.linalg.norm(alphahat - alpha_star)))
			width_curves[m] = float(np.mean(errs))
			
	return {"curves_data": out, "curves_json": out_json, "width": width_curves}


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	# Check for manual args for backward compat
	parser = argparse.ArgumentParser()
	parser.add_argument("--outdir", type=str, default="figures/sweeps")
	
	# Parse known args without erroring on hydra args
	args, _ = parser.parse_known_args()
	
	# Hydra config overrides or defaults
	outdir = cfg.get("outdir", args.outdir)
	os.makedirs(outdir, exist_ok=True)
	
	seeds = cfg.get("seeds", [123, 456, 789])
	# Use explicit lists if in config, else defaults
	lam_list = cfg.get("lam_list", [1e-1, 1e-2])
	t_list = cfg.get("t_list", [1, 2, 4, 6, 8])
	cond_list = cfg.get("cond_list", [10.0, 100.0])
	m_list = cfg.get("m_list", [8, 16, 24, 32])
	
	# Hydra returns OmegaConf ListConfig, convert to standard list
	seeds = list(seeds)
	lam_list = list(lam_list)
	t_list = list(t_list)
	cond_list = list(cond_list)
	m_list = list(m_list)

	res = run_sweep(
		seeds=seeds, 
		n=int(cfg.get("n_support", 64)), 
		p=int(cfg.get("p", 16)), 
		lam_list=lam_list, 
		t_list=t_list, 
		cond_list=cond_list, 
		m_list=m_list
	)
	
	out = res["curves_data"]
	width_curves = res["width"]
	
	try:
		import matplotlib.pyplot as plt
		for (cond, lam), curve in out.items():
			ts = [t for (t, e) in curve]
			es = [e for (t, e) in curve]
			plt.figure()
			plt.semilogy(ts, es, marker='o')
			plt.xlabel('t (CG steps)')
			plt.ylabel('||alpha_t - alpha*||')
			plt.title(f'CG rate vs t (cond={cond}, lam={lam})')
			plt.tight_layout()
			plt.savefig(os.path.join(outdir, f'cg_rate_cond{cond}_lam{lam}.png'), dpi=150)
			plt.close()
			
		# Width plot
		if width_curves:
			ms = sorted(width_curves.keys())
			vals = [width_curves[m] for m in ms]
			plt.figure()
			plt.plot(ms, vals, marker='o')
			plt.xlabel('width m')
			plt.ylabel('||alpha_hat - alpha*||')
			plt.title('Width–rank degradation')
			plt.tight_layout()
			plt.savefig(os.path.join(outdir, 'width_rank_curve.png'), dpi=150)
			plt.close()
	except Exception as e:
		print(f"Plotting failed: {e}")
		
	print(json.dumps({"curves": list(res["curves_json"].keys()), "width": width_curves}))


if __name__ == "__main__":
	main()
