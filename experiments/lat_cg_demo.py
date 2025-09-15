import numpy as np
from src.lat.cg_stack import run_cg


def main():
	rng = np.random.default_rng(42)
	n, p = 64, 16
	phi = rng.standard_normal((n, p)).astype(np.float64)
	K = phi @ phi.T
	lam = 1e-1
	y = rng.standard_normal(n).astype(np.float64)
	alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
	errs = []
	Ts = [1, 2, 4, 6, 8, 10]
	for t in Ts:
		alpha_t, _, _ = run_cg(phi, y, lam, t)
		errs.append(float(np.linalg.norm(alpha_t - alpha_star)))
	print({"t": Ts, "err": errs})
	try:
		import matplotlib.pyplot as plt
		plt.figure()
		plt.semilogy(Ts, errs, marker='o')
		plt.xlabel('CG steps t')
		plt.ylabel('||alpha_t - alpha*||')
		plt.title('CG convergence')
		plt.tight_layout()
		plt.show()
	except Exception:
		pass


if __name__ == "__main__":
	main()
