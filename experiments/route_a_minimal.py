import json
import math
import argparse
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig
import hydra

# Patch argparse for Hydra/Py3.14
_orig_add_argument = argparse.ArgumentParser.add_argument
def _safe_add_argument(self, *args, **kwargs):
	if 'help' in kwargs:
		h = kwargs['help']
		if hasattr(h, '__class__') and h.__class__.__name__ == 'LazyCompletionHelp':
			kwargs['help'] = "Shell completion"
	return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _safe_add_argument

def run_route_a_minimal(
	seed: int = 123,
	n_support: int = 48,
	n_query: int = 32,
	p: int = 16,
	d_proj: int = 12,
	tau: float = 0.5,
	lam: float = 1e-2,
	noise: float = 0.1,
) -> Dict[str, float]:
	torch.manual_seed(seed)
	rng = np.random.default_rng(seed)
	# Features φ for supports and queries
	phi_s = torch.randn(n_support, p, dtype=torch.float64)
	phi_q = torch.randn(n_query, p, dtype=torch.float64)
	# Random projection U and temperature τ
	U = torch.randn(d_proj, p, dtype=torch.float64) / math.sqrt(p)
	Qs = (U @ phi_s.T).T  # (k, d)
	Qq = (U @ phi_q.T).T  # (nq, d)
	# Exponential kernel on supports and queries
	S_ss = (Qs @ Qs.T) / tau
	K_exp_ss = torch.exp(S_ss)  # (k, k)
	S_sq = (Qs @ Qq.T) / tau
	K_exp_sq = torch.exp(S_sq)  # (k, nq)
	# Oracle KRR on exponential kernel
	y_s = torch.randn(n_support, dtype=torch.float64)
	y_s = y_s + noise * torch.randn_like(y_s)
	A = K_exp_ss + lam * torch.eye(n_support, dtype=torch.float64)
	alpha = torch.linalg.solve(A, y_s)
	f_oracle = (K_exp_sq.T @ alpha)  # (nq,)
	# Softmax smoother baseline (normalized kernel smoother)
	W = torch.softmax(S_sq, dim=0)  # normalize across supports for each query
	f_softmax = (W.T @ y_s)
	# Operator-norm proximity of support kernels
	softmax_rows = torch.softmax(S_ss, dim=1)
	row_sums = torch.sum(torch.exp(S_ss), dim=1, keepdim=True)  # Z_i per row
	K_from_softmax = softmax_rows * row_sums  # reconstruct exp(S_ss)
	op_norm_diff = torch.linalg.norm(K_from_softmax - K_exp_ss, ord=2).item()
	# Targets for evaluation: use a linear function of φ_q for ground-truth y
	w_true = torch.randn(p, dtype=torch.float64)
	y_q_true = (phi_q @ w_true) + noise * torch.randn(n_query, dtype=torch.float64)
	def rmse(pred: torch.Tensor) -> float:
		return float(torch.sqrt(torch.mean((pred - y_q_true) ** 2)))
	return {
		"rmse_oracle": rmse(f_oracle),
		"rmse_softmax": rmse(f_softmax),
		"rmse_gap": abs(rmse(f_oracle) - rmse(f_softmax)),
		"op_norm_diff": float(op_norm_diff),
		"tau": float(tau),
		"lambda": float(lam),
	}


@hydra.main(config_path="../configs", config_name="route_a", version_base=None)
def main(cfg: DictConfig) -> None:
	# For plotting, we check sys.argv manually or use hydra's cfg if we added a plot flag to config
	# But here we use argparse within hydra main which is unconventional but works if we use parse_known_args
	parser = argparse.ArgumentParser()
	parser.add_argument("--plot", action="store_true")
	parser.add_argument("--out", type=str, default="figures/route_a_mvp.png")
	# We only parse the specific args we added, ignoring hydra's args
	args, _ = parser.parse_known_args()
	
	res = run_route_a_minimal(
		seed=int(cfg.get("seed", 123)),
		n_support=int(cfg.get("n_support", 48)),
		n_query=int(cfg.get("n_query", 32)),
		p=int(cfg.get("p", 16)),
		d_proj=int(cfg.get("d_proj", 12)),
		tau=float(cfg.get("tau", 0.5)),
		lam=float(cfg.get("lambda", 1e-2)),
		noise=float(cfg.get("noise", 0.1)),
	)
	print(json.dumps(res))
	
	if cfg.get("plot", False):
		out_path = cfg.get("out", "figures/route_a_mvp.png")
		try:
			import matplotlib.pyplot as plt
			# simple bar plot of RMSEs
			vals = [res["rmse_oracle"], res["rmse_softmax"]]
			labels = ["oracle", "softmax"]
			plt.figure()
			plt.bar(labels, vals)
			plt.title("Route A MVP RMSE")
			plt.tight_layout()
			import os
			os.makedirs(os.path.dirname(out_path), exist_ok=True)
			plt.savefig(out_path, dpi=150)
		except Exception:
			pass


if __name__ == "__main__":
	main()
