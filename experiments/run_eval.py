import sys
from pathlib import Path
import json
import subprocess
from typing import List

# Add repo root to path to allow 'src' imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

import argparse
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

# Now we can import from experiments/ and src/
# When running python experiments/run_eval.py, experiments/ is in path, so we can import baselines directly
try:
	from experiments.baselines.ridge_oracle import run_ridge_oracle
	from experiments.baselines.gd_icl import run_gd_icl
	from experiments.route_a_minimal import run_route_a_minimal
	from experiments.width_rank import run_width_rank
	from experiments.precond import run_precond
	from experiments.route_a_end2end import route_a_end2end
	from experiments.route_b_approx import run_cg_route_b
	from experiments.probes.state_probes import run_probes
	from experiments.real_data_eval import run_real_data_eval
except ImportError:
	from baselines.ridge_oracle import run_ridge_oracle
	from baselines.gd_icl import run_gd_icl
	from route_a_minimal import run_route_a_minimal
	from width_rank import run_width_rank
	from precond import run_precond
	from route_a_end2end import route_a_end2end
	from route_b_approx import run_cg_route_b
	from probes.state_probes import run_probes
	from real_data_eval import run_real_data_eval

from src.eval.metrics import mean_ci


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	parser = argparse.ArgumentParser()
	# Define args for help message generation and manual parsing fallback
	parser.add_argument("--target", type=str, default="baselines", 
		choices=["baselines", "width_rank", "route_a", "precond", "end2end", "route_b", "probes", "real_data"])
	parser.add_argument("--plot", action="store_true")
	
	# Parse known args for backward compatibility if provided via CLI flags directly
	args, _ = parser.parse_known_args()

	target = cfg.get("target", args.target)
	plot = bool(cfg.get("plot", args.plot))
	metrics_cfg = cfg.get("metrics", {})
	alpha = float(metrics_cfg.get("alpha", 0.05))

	if target == "baselines":
		seeds = cfg.get("seeds", [123, 456, 789])
		ridge_rmses = []
		gd_rmses = []
		for s in seeds:
			r1 = run_ridge_oracle(
				n_support=cfg.get("n_support", 32),
				n_query=cfg.get("n_query", 32),
				p=cfg.get("p", 16),
				noise=cfg.get("noise", 0.1),
				lam=float(cfg.get("lambda", 1e-2)),
				seed=int(s)
			)
			r2 = run_gd_icl(
				n_support=cfg.get("n_support", 32),
				n_query=cfg.get("n_query", 32),
				p=cfg.get("p", 16),
				noise=cfg.get("noise", 0.1),
				steps=int(cfg.get("steps", 200)),
				lr=float(cfg.get("lr", 0.1)),
				lam=float(cfg.get("lambda", 1e-2)),
				seed=int(s)
			)
			ridge_rmses.append(float(r1["rmse"]))
			gd_rmses.append(float(r2["rmse"]))
		print({
			"ridge": mean_ci(ridge_rmses, alpha=alpha),
			"gd_icl": mean_ci(gd_rmses, alpha=alpha),
		})
	elif target == "width_rank":
		# Use proper function call
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
				ms = res["m"]
				errs = res["pred_err"]
				plt.figure()
				plt.plot(ms, errs, marker='o')
				plt.xlabel('width m (rank of sketch)')
				plt.ylabel('prediction error vs oracle')
				plt.title('Width–rank empirical')
				plt.tight_layout()
				out_path = str(cfg.get("out", "figures/width_rank.png"))
				plt.savefig(out_path, dpi=150)
			except Exception:
				pass

	elif target == "route_a":
		seeds = cfg.get("seeds", [123, 456, 789])
		or_rmse = []
		sm_rmse = []
		gap = []
		op = []
		for s in seeds:
			res = run_route_a_minimal(
				seed=int(s),
				n_support=int(cfg.get("n_support", 48)),
				n_query=int(cfg.get("n_query", 32)),
				p=int(cfg.get("p", 16)),
				d_proj=int(cfg.get("d_proj", 12)),
				tau=float(cfg.get("tau", 0.5)),
				lam=float(cfg.get("lambda", 1e-2)),
				noise=float(cfg.get("noise", 0.1)),
			)
			or_rmse.append(float(res["rmse_oracle"]))
			sm_rmse.append(float(res["rmse_softmax"]))
			gap.append(float(res["rmse_gap"]))
			op.append(float(res["op_norm_diff"]))
		print({
			"rmse_oracle": mean_ci(or_rmse, alpha=alpha),
			"rmse_softmax": mean_ci(sm_rmse, alpha=alpha),
			"rmse_gap": mean_ci(gap, alpha=alpha),
			"op_norm_diff": mean_ci(op, alpha=alpha),
			"n": len(seeds),
		})
	elif target == "precond":
		res = run_precond(
			seed=int(cfg.get("seed", 123)),
			n=int(cfg.get("n_support", 128)),
			p=int(cfg.get("p", 32)),
			lam=float(cfg.get("lambda", 1e-2)),
			t=int(cfg.get("steps", 8)),
			cond=float(cfg.get("cond", 100.0)) # Default to ill-conditioned to show effect
		)
		print(json.dumps(res))
	elif target == "end2end":
		res = route_a_end2end(
			seed=int(cfg.get("seed", 123)),
			k=int(cfg.get("n_support", 32)),
			p=int(cfg.get("p", 16)),
			d_proj=int(cfg.get("d_proj", 12)),
			tau=float(cfg.get("tau", 0.5)),
			lam=float(cfg.get("lambda", 1e-2)),
		)
		print(json.dumps(res))
	elif target == "route_b":
		import numpy as np
		seed = int(cfg.get("seed", 123))
		rng = np.random.default_rng(seed)
		n = int(cfg.get("n_support", 64))
		p = int(cfg.get("p", 16))
		lam = float(cfg.get("lambda", 1e-2))
		epsilon = float(cfg.get("epsilon", 1e-4))
		
		phi = rng.standard_normal((n, p))
		phi -= phi.mean(axis=0)
		y = rng.standard_normal(n)
		y -= y.mean()
		
		res = run_cg_route_b(phi, y, lam, t=int(cfg.get("steps", 5)), epsilon=epsilon)
		print(json.dumps(res))
	elif target == "probes":
		res = run_probes(
			seed=int(cfg.get("seed", 123)),
			n=int(cfg.get("n_support", 64)),
			p=int(cfg.get("p", 16)),
			steps=int(cfg.get("steps", 4))
		)
		print(json.dumps(res))
	elif target == "real_data":
		cwd = Path(__file__).resolve().parent
		repo_root = cwd.parent
		data_path = repo_root / "data" / "sentiment.csv"
		
		if not data_path.exists():
			print(f"Error: Data file not found at {data_path}")
		else:
			res = run_real_data_eval(
				str(data_path),
				n_support=int(cfg.get("n_support", 8)),
				n_query=int(cfg.get("n_query", 4)),
				seed=int(cfg.get("seed", 123))
			)
			print(json.dumps(res))


if __name__ == "__main__":
	main()
