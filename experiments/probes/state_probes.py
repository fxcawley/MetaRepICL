import json
import numpy as np
from typing import Dict
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


def fit_linear_probe(X: np.ndarray, y: np.ndarray) -> float:
    # Closed-form least squares; return cosine similarity between predictions and target
    # Check dimensions
    # X: (N_samples, D_in)
    # y: (N_samples, D_out)
    
    # Guard against under-determined system (though here N > D)
    # Use ridge for stability
    n_samples, n_features = X.shape
    if n_samples < n_features:
        print(f"Warning: Undersampled probe fit ({n_samples} < {n_features})")
        
    # Ridge solve: W = (X^T X + lam I)^-1 X^T y
    lam = 1e-3
    XtX = X.T @ X
    reg = lam * np.eye(XtX.shape[0])
    W = np.linalg.solve(XtX + reg, X.T @ y)
    
    pred = X @ W
    pred_flat = pred.flatten()
    y_flat = y.flatten()
    
    norm_p = np.linalg.norm(pred_flat)
    norm_y = np.linalg.norm(y_flat)
    
    if norm_p < 1e-9 or norm_y < 1e-9:
        return 0.0
        
    cos = float(np.dot(pred_flat, y_flat) / (norm_p * norm_y))
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
