import json
import numpy as np
from src.lat.cg_stack import run_cg_with_history


def generate_cg_states(n: int = 64, p: int = 16, steps: int = 32, seed: int = 123):
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


def fit_linear_probe(
    X: np.ndarray,
    Y: np.ndarray,
    train_fraction: float = 0.75,
    ridge: float = 1e-3,
) -> float:
    """Fit a linear probe and evaluate cosine similarity on a holdout split.

    Args:
        X: Activation matrix with shape (steps, d_hidden).
        Y: Target matrix with shape (steps, d_target) or vector (steps,).
        train_fraction: Fraction of steps to use for fitting the probe.
        ridge: L2 regularization strength to stabilize the solve.

    Returns:
        Cosine similarity between vectorized predictions and ground-truth on the
        evaluation split.
    """

    if Y.ndim == 1:
        Y = Y[:, None]

    n_steps = X.shape[0]
    split = max(1, min(n_steps - 1, int(round(train_fraction * n_steps))))
    if split <= 0:
        split = max(1, n_steps - 1)

    X_train, Y_train = X[:split], Y[:split]
    X_eval, Y_eval = X[split:], Y[split:]

    if X_eval.shape[0] == 0:
        X_train, Y_train = X[:-1], Y[:-1]
        X_eval, Y_eval = X[-1:], Y[-1:]

    XtX = X_train.T @ X_train
    if ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0], dtype=XtX.dtype)
    W = np.linalg.solve(XtX, X_train.T @ Y_train)

    pred = X_eval @ W
    y_vec = Y_eval.reshape(-1)
    p_vec = pred.reshape(-1)
    denom = np.linalg.norm(p_vec) * np.linalg.norm(y_vec)
    if denom == 0:
        return 0.0
    return float((p_vec @ y_vec) / denom)


def main():
	states, acts_true, acts_ctrl = generate_cg_states()
	# Target: recover concatenated state
	target = []
	for (a, r, p) in states:
		target.append(np.concatenate([a, r, p]))
	target = np.array(target)
	cos_true = fit_linear_probe(acts_true, target)
	cos_ctrl = fit_linear_probe(acts_ctrl, target)
	print(json.dumps({"cos_true": cos_true, "cos_control": cos_ctrl}))


if __name__ == "__main__":
	main()
