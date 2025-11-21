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
except ImportError:
	from baselines.ridge_oracle import run_ridge_oracle
	from baselines.gd_icl import run_gd_icl
	from route_a_minimal import run_route_a_minimal

from src.eval.metrics import mean_ci


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	target = cfg.get("target", "baselines")
	plot = bool(cfg.get("plot", False))
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
		# Legacy subprocess call for width_rank until refactored
		cmd = [sys.executable, str(repo_root / "experiments/width_rank.py")] + (["--plot"] if plot else [])
		out = subprocess.check_output(cmd, cwd=str(repo_root))
		print(json.loads(out.decode('utf-8').replace("'", '"')))
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


if __name__ == "__main__":
	main()
