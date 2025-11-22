import json
import math
import argparse
from typing import Dict, Any

import numpy as np
import torch
from omegaconf import DictConfig
import hydra

# Patch argparse for Hydra/Py3.14
_orig_add_argument = argparse.ArgumentParser.add_argument
def _safe_add_argument(self, *args, **kwargs):
    if 'help' in kwargs:
        h = kwargs['help']
        if hasattr(h, '__class__') and h.__class__.__name__ == 'LazyCompletionHelp':
            kwargs['help'] = "Shell completion"
    return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _safe_add_argument

def run_route_a_minimal(
    seed: int = 123,
    n_support: int = 48,
    n_query: int = 32,
    p: int = 16,
    d_proj: int = 12,
    tau: float = 0.5,
    lam: float = 1e-2,
    noise: float = 0.1,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    
    # Define ground truth function
    w_true = torch.randn(p, dtype=torch.float64)

    # Features φ for supports and queries
    phi_s = torch.randn(n_support, p, dtype=torch.float64)
    phi_q = torch.randn(n_query, p, dtype=torch.float64)
    
    # Targets for evaluation: use a linear function of φ for ground-truth y
    y_s = (phi_s @ w_true) + noise * torch.randn(n_support, dtype=torch.float64)
    y_q_true = (phi_q @ w_true) + noise * torch.randn(n_query, dtype=torch.float64)

    # Random projection U and temperature τ
    U = torch.randn(d_proj, p, dtype=torch.float64) / math.sqrt(p)
    Qs = (U @ phi_s.T).T  # (k, d)
    Qq = (U @ phi_q.T).T  # (nq, d)
    
    # Exponential kernel on supports and queries
    S_ss = (Qs @ Qs.T) / tau
    K_exp_ss = torch.exp(S_ss)  # (k, k)
    
    S_sq = (Qs @ Qq.T) / tau
    K_exp_sq = torch.exp(S_sq)  # (k, nq)
    
    # Oracle KRR on exponential kernel
    A = K_exp_ss + lam * torch.eye(n_support, dtype=torch.float64)
    alpha = torch.linalg.solve(A, y_s)
    f_oracle = (K_exp_sq.T @ alpha)  # (nq,)
    
    # Softmax smoother baseline (normalized kernel smoother)
    W = torch.softmax(S_sq, dim=0)  # normalize across supports for each query
    f_softmax = (W.T @ y_s)
    
    # Operator-norm proximity of support kernels
    softmax_rows = torch.softmax(S_ss, dim=1)
    row_sums = torch.sum(torch.exp(S_ss), dim=1, keepdim=True)  # Z_i per row
    K_from_softmax = softmax_rows * row_sums  # reconstruct exp(S_ss)
    op_norm_diff = torch.linalg.norm(K_from_softmax - K_exp_ss, ord=2).item()

    # --- Deep Transformer (GD-KRR) ---
    # Simulate L layers of Gradient Descent using the Softmax operator.
    # Goal: Solve (K + lam I) alpha = y
    # Update: alpha <- alpha - eta * ((K + lam I) alpha - y)
    # Operator: v -> K v is approximated by (Softmax(QK^T) * Z) v
    
    alpha_deep = torch.zeros(n_support, dtype=torch.float64)
    
    # Use the exact kernel reconstruction from Softmax to verify Algorithm capacity
    K_op = K_from_softmax
    
    # Adaptive Learning Rate for stability
    # Max eigenvalue of K_op
    max_eig = torch.linalg.matrix_norm(K_op, ord=2).item()
    # Optimal rate for convex problem is 2/L, but with noise/approx safer to use 1/L
    # We need to account for lambda too: L = max_eig + lam
    L_smooth = max_eig + lam
    eta = 1.0 / L_smooth
    
    steps = 50 # Depth of transformer
    
    for _ in range(steps):
        # 1. Compute K * alpha using the attention mechanism approximation
        k_alpha = K_op @ alpha_deep
        
        # 2. Compute gradient: (K alpha + lam alpha - y)
        grad = k_alpha + lam * alpha_deep - y_s
        
        # 3. Update alpha (Residual connection with MLP)
        alpha_deep = alpha_deep - eta * grad
        
    # Readout
    f_deep = (K_exp_sq.T @ alpha_deep)

    def rmse(pred: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean((pred - y_q_true) ** 2)))
        
    return {
        "rmse_oracle": rmse(f_oracle),
        "rmse_softmax": rmse(f_softmax),
        "rmse_deep": rmse(f_deep),
        "rmse_gap": abs(rmse(f_oracle) - rmse(f_softmax)),
        "rmse_deep_gap": abs(rmse(f_oracle) - rmse(f_deep)),
        "op_norm_diff": float(op_norm_diff),
        "tau": float(tau),
        "lambda": float(lam),
        # Return vectors for plotting
        "f_oracle": f_oracle.tolist(),
        "f_softmax": f_softmax.tolist(),
        "f_deep": f_deep.tolist(),
        "y_q_true": y_q_true.tolist(),
    }


@hydra.main(config_path="../configs", config_name="route_a", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", type=str, default="docs/figures/route_a_mvp.png")
    args, _ = parser.parse_known_args()
    
    res = run_route_a_minimal(
        seed=int(cfg.get("seed", 123)),
        n_support=int(cfg.get("n_support", 48)),
        n_query=int(cfg.get("n_query", 32)),
        p=int(cfg.get("p", 16)),
        d_proj=int(cfg.get("d_proj", 12)),
        tau=float(cfg.get("tau", 2.0)), # Use 2.0 stable default
        lam=float(cfg.get("lambda", 1e-2)),
        noise=float(cfg.get("noise", 0.1)),
    )
    
    # Print JSON results without the large vectors
    print_res = {k: v for k, v in res.items() if k not in ["f_oracle", "f_softmax", "f_deep", "y_q_true"]}
    print(json.dumps(print_res))
    
    if cfg.get("plot", False):
        out_path = cfg.get("out", "docs/figures/route_a_mvp.png")
        try:
            import matplotlib.pyplot as plt
            import os
            
            f_oracle = np.array(res["f_oracle"])
            f_softmax = np.array(res["f_softmax"])
            f_deep = np.array(res["f_deep"])
            
            # Sort by Oracle value for clarity
            idxs = np.argsort(f_oracle)
            
            plt.figure(figsize=(8, 5))
            plt.plot(f_oracle[idxs], label='Oracle (KRR)', color='blue', marker='o', markersize=4, linewidth=1.5)
            plt.plot(f_deep[idxs], label='Deep Transformer (GD)', color='green', marker='s', markersize=3, linestyle='-')
            plt.plot(f_softmax[idxs], label='1-Layer (Softmax)', color='orange', marker='x', markersize=4, linestyle='--', alpha=0.5)
            
            plt.title("Route A: Deep Transformer Construction (Green) vs Oracle (Blue)")
            plt.xlabel("Query Sample (sorted by Oracle value)")
            plt.ylabel("Prediction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot to {out_path}")
        except Exception as e:
            print(f"Error plotting: {e}")


if __name__ == "__main__":
    main()
