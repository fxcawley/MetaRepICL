import json
import numpy as np
from src.lat.cg_stack import run_cg_with_history


def generate_cg_dataset(num_tasks: int = 100, n: int = 64, p: int = 16, steps: int = 4, seed: int = 123):
    rng = np.random.default_rng(seed)
    all_activations_true = []
    all_activations_control = []
    all_targets = []
    
    W_true = rng.standard_normal((3 * n, 3 * n)) # Fixed probe projection for "true" model
    
    for i in range(num_tasks):
        # Different problem per task
        phi = rng.standard_normal((n, p)).astype(np.float64)
        y = rng.standard_normal(n).astype(np.float64)
        hist = run_cg_with_history(phi, y, lam=1e-1, t=steps)
        
        for (a, r_, p_) in hist:
            z = np.concatenate([a, r_, p_])
            all_targets.append(z)
            # True: Linear transform of state
            all_activations_true.append(W_true @ z)
            # Control: Random noise (fixed dimension)
            all_activations_control.append(rng.standard_normal(3 * n))
            
    return np.array(all_targets), np.array(all_activations_true), np.array(all_activations_control)


<<<<<<< HEAD
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
=======
def fit_linear_probe(X: np.ndarray, y: np.ndarray) -> float:
    # Closed-form least squares; return cosine similarity between predictions and target
    # Check dimensions
    # X: (N_samples, D_in)
    # y: (N_samples, D_out)
    W = np.linalg.pinv(X) @ y
    pred = X @ W
    pred_flat = pred.flatten()
    y_flat = y.flatten()
    cos = float(np.dot(pred_flat, y_flat) / (np.linalg.norm(pred_flat) * np.linalg.norm(y_flat) + 1e-18))
    return cos


def run_probes(seed: int = 123, n: int = 64, p: int = 16, steps: int = 4) -> Dict[str, float]:
    # Generate enough data to avoid overfitting
    # Dim = 3*n = 192. Need > 192 samples.
    # 200 tasks * 4 steps = 800 samples.
    targets, acts_true, acts_ctrl = generate_cg_dataset(num_tasks=200, n=n, p=p, steps=steps, seed=seed)
    
    cos_true = fit_linear_probe(acts_true, targets)
    cos_ctrl = fit_linear_probe(acts_ctrl, targets)
    return {"cos_true": cos_true, "cos_control": cos_ctrl}


def main():
	res = run_probes()
	print(json.dumps(res))


if __name__ == "__main__":
	main()
