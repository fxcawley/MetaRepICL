import json
import numpy as np
from typing import Dict


def generate_cg_states(n: int = 64, p: int = 16, steps: int = 4, seed: int = 123):
	rng = np.random.default_rng(seed)
	phi = rng.standard_normal((n, p)).astype(np.float64)
	K = phi @ phi.T
	y = rng.standard_normal(n).astype(np.float64)
	alpha = np.zeros(n, dtype=np.float64)
	r = y.copy()
	pvec = r.copy()
	states = []
	for _ in range(steps):
		Ap = K @ pvec + 1e-1 * pvec
		rr = float(r @ r)
		pAp = float(pvec @ Ap)
		gamma = rr / (pAp + 1e-18)
		alpha = alpha + gamma * pvec
		r = r - gamma * Ap
		rr_next = float(r @ r)
		beta = rr_next / (rr + 1e-18)
		pvec = r + beta * pvec
		states.append((alpha.copy(), r.copy(), pvec.copy()))
	# Fake activations: true embedding carries a linear map of (alpha,r,p);
	# control embedding is random unrelated noise
	W_true = rng.standard_normal((3 * n, 3 * n))
	activations_true = []
	activations_control = []
	for (a, r_, p_) in states:
		z = np.concatenate([a, r_, p_])
		activations_true.append(W_true @ z)
		activations_control.append(rng.standard_normal(3 * n))
	return states, np.array(activations_true), np.array(activations_control)


def fit_linear_probe(X: np.ndarray, y: np.ndarray) -> float:
	# Closed-form least squares; return cosine similarity between predictions and target
	W = np.linalg.pinv(X) @ y
	pred = X @ W
	cos = float((pred @ y) / (np.linalg.norm(pred) * np.linalg.norm(y) + 1e-18))
	return cos


def main():
	states, acts_true, acts_ctrl = generate_cg_states()
	# Target: recover concatenated state
	target = []
	for (a, r, p) in states:
		target.append(np.concatenate([a, r, p]))
	target = np.array(target)
	cos_true = fit_linear_probe(acts_true, target.flatten())
	cos_ctrl = fit_linear_probe(acts_ctrl, target.flatten())
	print(json.dumps({"cos_true": cos_true, "cos_control": cos_ctrl}))


if __name__ == "__main__":
	main()
