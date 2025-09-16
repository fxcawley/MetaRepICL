import argparse
import subprocess
import json
from typing import List
from omegaconf import DictConfig
import hydra
from src.eval.metrics import mean_ci


def run_cmd(cmd: List[str]) -> dict:
	out = subprocess.check_output(cmd)
	return json.loads(out.decode('utf-8').replace("'", '"'))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, default="baselines", choices=["baselines", "width_rank", "route_a"])
	parser.add_argument("--plot", action="store_true")
	args, _ = parser.parse_known_args()

	if args.target == "baselines":
		seeds = cfg.get("seeds", [123, 456, 789])
		ridge_rmses = []
		gd_rmses = []
		for s in seeds:
			r1 = run_cmd(["python", "experiments/baselines/ridge_oracle.py", f"seed={s}"])
			r2 = run_cmd(["python", "experiments/baselines/gd_icl.py", f"seed={s}"])
			ridge_rmses.append(float(r1["rmse"]))
			gd_rmses.append(float(r2["rmse"]))
		print({
			"ridge": mean_ci(ridge_rmses),
			"gd_icl": mean_ci(gd_rmses),
		})
	elif args.target == "width_rank":
		cmd = ["python", "experiments/width_rank.py"] + (["--plot"] if args.plot else [])
		res = run_cmd(cmd)
		print(res)
	elif args.target == "route_a":
		res = run_cmd(["python", "experiments/route_a_minimal.py"])
		print(res)


if __name__ == "__main__":
	main()
