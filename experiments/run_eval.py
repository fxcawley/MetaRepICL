import argparse
import subprocess
import json
from typing import List
from pathlib import Path
import sys
from omegaconf import DictConfig
import hydra
from src.eval.metrics import mean_ci



def run_cmd(cmd: List[str], cwd: Path) -> dict:
	out = subprocess.check_output(cmd, cwd=str(cwd))
	return json.loads(out.decode('utf-8').replace("'", '"'))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, default="baselines", choices=["baselines", "width_rank", "route_a"])
	parser.add_argument("--plot", action="store_true")
	args, _ = parser.parse_known_args()

	# Resolve absolute paths relative to this file's repo root
	file_path = Path(__file__).resolve()
	repo_root = file_path.parent.parent  # MetaRepICL/

	if args.target == "baselines":
		seeds = cfg.get("seeds", [123, 456, 789])
		ridge_rmses = []
		gd_rmses = []
		for s in seeds:
			r1 = run_cmd([sys.executable, str(repo_root / "experiments/baselines/ridge_oracle.py"), f"seed={s}"], cwd=repo_root)
			r2 = run_cmd([sys.executable, str(repo_root / "experiments/baselines/gd_icl.py"), f"seed={s}"], cwd=repo_root)
			ridge_rmses.append(float(r1["rmse"]))
			gd_rmses.append(float(r2["rmse"]))
		print({
			"ridge": mean_ci(ridge_rmses),
			"gd_icl": mean_ci(gd_rmses),
		})
	elif args.target == "width_rank":
		cmd = [sys.executable, str(repo_root / "experiments/width_rank.py")] + (["--plot"] if args.plot else [])
		res = run_cmd(cmd, cwd=repo_root)
		print(res)
	elif args.target == "route_a":
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
			"rmse_oracle": mean_ci(or_rmse),
			"rmse_softmax": mean_ci(sm_rmse),
			"rmse_gap": mean_ci(gap),
			"op_norm_diff": mean_ci(op),
			"n": len(seeds),
		})


if __name__ == "__main__":
	main()
