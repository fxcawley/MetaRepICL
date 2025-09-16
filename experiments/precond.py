import json
import numpy as np
from src.lat.cg_stack import run_cg


def run_precond(seed: int = 123, n: int = 128, p: int = 32, lam: float = 1e-2, t: int = 8):
	rng = np.random.default_rng(seed)
	phi = rng.standard_normal((n, p)).astype(np.float64)
	K = phi @ phi.T
	y = rng.standard_normal(n).astype(np.float64)
	alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
	# Unpreconditioned
	errs_un = []
	alpha, r, pvec = run_cg(phi, y, lam, 0)
	for step in range(1, t + 1):
		alpha, r, pvec = run_cg(phi, y, lam, step)
		errs_un.append(float(np.linalg.norm(alpha - alpha_star)))
	# Diagonal preconditioner: P^{-1} = 1/(diag(K)+λ)
	diagA = np.diag(K) + lam
	pinv = 1.0 / (diagA + 1e-6)
	# Simple right-preconditioned CG by scaling inputs (proxy for illustration)
	y_tilde = pinv * y
	K_tilde = (pinv[:, None]) * K
	alpha_star_pre = np.linalg.solve(K_tilde + lam * np.eye(n), y_tilde)
	errs_pr = []
	for step in range(1, t + 1):
		# reuse same steps count as proxy; in practice integrate P^{-1} into CG step
		alpha_t, _, _ = run_cg(phi, y_tilde, lam, step)
		errs_pr.append(float(np.linalg.norm(alpha_t - alpha_star_pre)))
	return {"err_un": errs_un, "err_pr": errs_pr}


def main():
	res = run_precond()
	print(json.dumps(res))


if __name__ == "__main__":
	main()
