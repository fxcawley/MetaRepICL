import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.lat.cg_stack import run_cg
from experiments.width_rank import sketch_kernel, effective_dimension


def make_phi(n: int, p: int, seed: int, cond: float) -> np.ndarray:
	rng = np.random.default_rng(seed)
	U, _ = np.linalg.qr(rng.standard_normal((p, p)))
	s = np.geomspace(cond, 1.0, p)
	phi_base = rng.standard_normal((n, p))
	phi = phi_base @ (U * s)
	return phi


def sweep(
	seeds: List[int], n: int, p: int, lam_list: List[float], t_list: List[int], cond_list: List[float], m_list: List[int]
):
	out = {}
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
	return out, width_curves


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--outdir", type=str, default="figures/sweeps")
	args = parser.parse_args()
	os.makedirs(args.outdir, exist_ok=True)
	seeds = [123, 456, 789]
	lam_list = [1e-1, 1e-2]
	t_list = [1, 2, 4, 6, 8]
	cond_list = [10.0, 100.0]
	m_list = [8, 16, 24, 32]
	out, width_curves = sweep(seeds, n=64, p=16, lam_list=lam_list, t_list=t_list, cond_list=cond_list, m_list=m_list)
	for (cond, lam), curve in out.items():
		ts = [t for (t, e) in curve]
		es = [e for (t, e) in curve]
		plt.figure()
		plt.semilogy(ts, es, marker='o')
		plt.xlabel('t (CG steps)')
		plt.ylabel('||alpha_t - alpha*||')
		plt.title(f'CG rate vs t (cond={cond}, lam={lam})')
		plt.tight_layout()
		plt.savefig(os.path.join(args.outdir, f'cg_rate_cond{cond}_lam{lam}.png'), dpi=150)
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
		plt.savefig(os.path.join(args.outdir, 'width_rank_curve.png'), dpi=150)
	print(json.dumps({"curves": list(out.keys()), "width": width_curves}))


if __name__ == "__main__":
	main()
