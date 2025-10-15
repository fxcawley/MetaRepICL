import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig



def run_cmd(cmd: List[str], cwd: Path) -> dict:
	out = subprocess.check_output(cmd, cwd=str(cwd))
	return json.loads(out.decode('utf-8').replace("'", '"'))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

	# Resolve absolute paths relative to this file's repo root
	file_path = Path(__file__).resolve()
	repo_root = file_path.parent.parent  # MetaRepICL/
	# Ensure repo root is on sys.path for 'src' namespace imports
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	# Import after sys.path adjustment
	from src.eval.metrics import mean_ci  # type: ignore

	target = cfg.get("target", "baselines")
	plot = bool(cfg.get("plot", False))
	metrics_cfg = cfg.get("metrics", {})
	alpha = float(metrics_cfg.get("alpha", 0.05))
	use_t = bool(metrics_cfg.get("use_t", True))

	if target == "baselines":
		seeds = cfg.get("seeds", [123, 456, 789])
		ridge_rmses = []
		gd_rmses = []
		for s in seeds:
			r1 = run_cmd([sys.executable, str(repo_root / "experiments/baselines/ridge_oracle.py"), f"seed={s}"], cwd=repo_root)
			r2 = run_cmd([sys.executable, str(repo_root / "experiments/baselines/gd_icl.py"), f"seed={s}"], cwd=repo_root)
			ridge_rmses.append(float(r1["rmse"]))
			gd_rmses.append(float(r2["rmse"]))
		print({
			"ridge": mean_ci(ridge_rmses, alpha=alpha, use_t=use_t),
			"gd_icl": mean_ci(gd_rmses, alpha=alpha, use_t=use_t),
		})
	elif target == "width_rank":
		cmd = [sys.executable, str(repo_root / "experiments/width_rank.py")] + (["--plot"] if plot else [])
		res = run_cmd(cmd, cwd=repo_root)
		print(res)
	elif target == "route_a":
		seeds = cfg.get("seeds", [123, 456, 789])
		or_rmse = []
		sm_rmse = []
		gap = []
		op = []
		for s in seeds:
			res = run_cmd([sys.executable, str(repo_root / "experiments/route_a_minimal.py"), f"seed={int(s)}"], cwd=repo_root) 
			or_rmse.append(float(res["rmse_oracle"]))
			sm_rmse.append(float(res["rmse_softmax"]))
			gap.append(float(res["rmse_gap"]))
			op.append(float(res["op_norm_diff"]))
		print({
			"rmse_oracle": mean_ci(or_rmse, alpha=alpha, use_t=use_t),
			"rmse_softmax": mean_ci(sm_rmse, alpha=alpha, use_t=use_t),
			"rmse_gap": mean_ci(gap, alpha=alpha, use_t=use_t),
			"op_norm_diff": mean_ci(op, alpha=alpha, use_t=use_t),
			"n": len(seeds),
		})


if __name__ == "__main__":
	main()
