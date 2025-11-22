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

def fit_linear_probe(X: np.ndarray, y: np.ndarray) -> float:
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
    
    pred = X @ W
    pred_flat = pred.flatten()
    y_flat = y.flatten()
    
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
            
    # Fit probes
    layers = list(range(1, steps + 1))
    sims_true = []
    sims_ctrl = []
    
    for l in layers:
        X_true = np.array(acts_true_by_layer[l])
        X_ctrl = np.array(acts_ctrl_by_layer[l])
        Y = np.array(targets_by_layer[l])
        
        sims_true.append(fit_linear_probe(X_true, Y))
        sims_ctrl.append(fit_linear_probe(X_ctrl, Y))
        
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
