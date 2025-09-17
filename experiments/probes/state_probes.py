import json
import numpy as np
from typing import Dict
from src.lat.cg_stack import run_cg_with_history


def generate_cg_states(n: int = 64, p: int = 16, steps: int = 4, seed: int = 123):
	rng = np.random.default_rng(seed)
	phi = rng.standard_normal((n, p)).astype(np.float64)
	y = rng.standard_normal(n).astype(np.float64)
	hist = run_cg_with_history(phi, y, lam=1e-1, t=steps)
	states = hist
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
