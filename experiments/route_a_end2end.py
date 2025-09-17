import json
import math
import numpy as np
import torch


def route_a_end2end(seed: int = 123, k: int = 32, p: int = 16, d_proj: int = 12, tau: float = 0.5, lam: float = 1e-2):
	torch.manual_seed(seed)
	rng = np.random.default_rng(seed)
	# features and targets for supports
	phi_s = torch.randn(k, p, dtype=torch.float64)
	phi_q = torch.randn(1, p, dtype=torch.float64)  # single query
	U = torch.randn(d_proj, p, dtype=torch.float64) / math.sqrt(p)
	Qs = (U @ phi_s.T).T
	Qq = (U @ phi_q.T).T  # (1, d)
	S_ss = (Qs @ Qs.T) / tau
	S_sq = (Qs @ Qq.T) / tau  # (k, 1)
	K_exp_ss = torch.exp(S_ss)
	K_exp_sq = torch.exp(S_sq)  # (k, 1)
	y_s = torch.randn(k, dtype=torch.float64)
	A = K_exp_ss + lam * torch.eye(k, dtype=torch.float64)
	alpha = torch.linalg.solve(A, y_s)
	f_query = (K_exp_sq.T @ alpha).squeeze(0)
	# Ground truth from a linear function of φ_q
	w_true = torch.randn(p, dtype=torch.float64)
	y_true = (phi_q @ w_true).squeeze(0)
	return {
		"pred": float(f_query.item()),
		"y_true": float(y_true.item()),
		"abs_err": float(abs(f_query.item() - y_true.item())),
		"tau": float(tau),
		"lambda": float(lam),
	}


def main():
	res = route_a_end2end()
	print(json.dumps(res))


if __name__ == "__main__":
	main()
