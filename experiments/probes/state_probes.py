import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.lat.cg_stack import run_cg_with_history


def generate_cg_dataset(
    num_tasks: int = 20,
    n: int = 8,
    p: int = 4,
    steps: int = 10,
    seed: int = 321,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate CG state dataset for probe evaluation.

    Returns (targets, acts_true, acts_ctrl) where:
      - targets: (num_tasks * steps, 3*n) concatenated CG state vectors [alpha; r; p]
      - acts_true: (num_tasks * steps, 3*n) linear projection of true states
      - acts_ctrl: (num_tasks * steps, 3*n) random noise (negative control)
    """
    rng = np.random.default_rng(seed)
    W_true = rng.standard_normal((3 * n, 3 * n))

    targets_list: List[np.ndarray] = []
    acts_true_list: List[np.ndarray] = []
    acts_ctrl_list: List[np.ndarray] = []

    for _ in range(num_tasks):
        phi = rng.standard_normal((n, p)).astype(np.float64)
        y = rng.standard_normal(n).astype(np.float64)
        hist = run_cg_with_history(phi, y, lam=1e-1, t=steps)

        for a, r_, p_ in hist:
            z = np.concatenate([a, r_, p_])
            targets_list.append(z)
            acts_true_list.append(W_true @ z)
            acts_ctrl_list.append(rng.standard_normal(3 * n))

    targets = np.array(targets_list)
    acts_true = np.array(acts_true_list)
    acts_ctrl = np.array(acts_ctrl_list)
    return targets, acts_true, acts_ctrl


def fit_linear_probe(X: np.ndarray, y: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None) -> float:
    # Closed-form least squares; return cosine similarity between predictions and target
    # X: (N_samples, D_in)
    # y: (N_samples, D_out)
    
    n_samples, n_features = X.shape
    
    # Ridge solve: W = (X^T X + lam I)^-1 X^T y
    lam = 1e-3
    XtX = X.T @ X
    reg = lam * np.eye(XtX.shape[0])
    try:
        W = np.linalg.solve(XtX + reg, X.T @ y)
    except np.linalg.LinAlgError:
        return 0.0
    
    # Evaluate on test set if provided, otherwise train set
    X_eval = X_test if X_test is not None else X
    y_eval = y_test if y_test is not None else y
    pred = X_eval @ W
    pred_flat = pred.flatten()
    y_flat = y_eval.flatten()
    
    norm_p = np.linalg.norm(pred_flat)
    norm_y = np.linalg.norm(y_flat)
    
    if norm_p < 1e-9 or norm_y < 1e-9:
        return 0.0
        
    cos = float(np.dot(pred_flat, y_flat) / (norm_p * norm_y))
    return cos

def run_probes_per_layer(seed: int = 123, n: int = 64, p: int = 16, steps: int = 6):
    rng = np.random.default_rng(seed)
    W_true = rng.standard_normal((3 * n, 3 * n)) # Fixed probe projection for "true" model
    
    targets_by_layer = {l: [] for l in range(1, steps + 1)}
    acts_true_by_layer = {l: [] for l in range(1, steps + 1)}
    acts_ctrl_by_layer = {l: [] for l in range(1, steps + 1)}
    
    # Generate data
    num_tasks = 100
    for _ in range(num_tasks):
        phi = rng.standard_normal((n, p)).astype(np.float64)
        y = rng.standard_normal(n).astype(np.float64)
        hist = run_cg_with_history(phi, y, lam=1e-1, t=steps)
        
        for i, (a, r_, p_) in enumerate(hist):
            layer_idx = i + 1
            z = np.concatenate([a, r_, p_])
            
            targets_by_layer[layer_idx].append(z)
            acts_true_by_layer[layer_idx].append(W_true @ z)
            acts_ctrl_by_layer[layer_idx].append(rng.standard_normal(3 * n))
            
    # Fit probes with train/test split
    layers = list(range(1, steps + 1))
    sims_true = []
    sims_ctrl = []
    
    for l in layers:
        X_true = np.array(acts_true_by_layer[l])
        X_ctrl = np.array(acts_ctrl_by_layer[l])
        Y = np.array(targets_by_layer[l])
        
        # 80/20 train/test split
        n_total = X_true.shape[0]
        n_train = max(1, int(0.8 * n_total))
        
        X_true_train, X_true_test = X_true[:n_train], X_true[n_train:]
        X_ctrl_train, X_ctrl_test = X_ctrl[:n_train], X_ctrl[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        
        sims_true.append(fit_linear_probe(X_true_train, Y_train, X_true_test, Y_test))
        sims_ctrl.append(fit_linear_probe(X_ctrl_train, Y_train, X_ctrl_test, Y_test))
        
    return layers, sims_true, sims_ctrl

def main():
    layers, sims_true, sims_ctrl = run_probes_per_layer()
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(layers, sims_true, marker='o', label='Constructed Model (Simulation)')
    plt.plot(layers, sims_ctrl, marker='x', linestyle='--', label='Random Control')
    
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Layer / Step")
    plt.ylabel("Cosine Similarity")
    plt.title("Theoretical Recoverability of CG States")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = "docs/figures/probes/cosine_sim_layer.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
