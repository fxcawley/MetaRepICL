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
    n_support: int = 24,
    n_query: int = 16,
    p: int = 8,
    d_proj: int = 6,
    tau: float = 10.0,
    lam: float = 1.0,
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

    # --- CG on Softmax Kernel (Route A + Route B solver) ---
    # Run Conjugate Gradient to solve (K + lam I) alpha = y, where K is
    # the exponential kernel reconstructed from softmax attention weights.
    # This validates the theory's central claim (Corollary A / prop.md):
    # CG applies to Route A with K -> K_exp, converging at the CG rate
    # ((sqrt(kappa)-1)/(sqrt(kappa)+1))^t rather than the GD rate
    # ((kappa-1)/(kappa+1))^t.
    #
    # NOTE: This is still a numerical demonstration on a pre-computed kernel,
    # not a trained transformer. See REVIEW_ISSUES.md (W1).
    
    K_op = K_from_softmax
    n = n_support
    cg_steps = 15  # fewer steps to highlight CG vs GD convergence difference

    # CG iteration (textbook CG on (K + lam I) alpha = y)
    alpha_cg = torch.zeros(n, dtype=torch.float64)
    r_cg = y_s.clone()
    p_cg = r_cg.clone()
    cg_errors = []
    for _ in range(cg_steps):
        Kp = K_op @ p_cg
        Ap = Kp + lam * p_cg
        rr = float(r_cg @ r_cg)
        pAp = float(p_cg @ Ap)
        gamma = rr / (pAp + 1e-18)
        alpha_cg = alpha_cg + gamma * p_cg
        r_cg = r_cg - gamma * Ap
        rr_new = float(r_cg @ r_cg)
        beta = rr_new / (rr + 1e-18)
        p_cg = r_cg + beta * p_cg
        # Track error to oracle solution
        cg_errors.append(float(torch.norm(alpha_cg - alpha)))

    f_cg = (K_exp_sq.T @ alpha_cg)

    # --- GD on Softmax Kernel (for comparison) ---
    # Standard gradient descent on the same kernel, same number of steps.
    # Expected to converge slower than CG, especially for ill-conditioned K.
    max_eig = torch.linalg.matrix_norm(K_op, ord=2).item()
    L_smooth = max_eig + lam
    eta = 1.0 / L_smooth

    alpha_gd = torch.zeros(n, dtype=torch.float64)
    gd_errors = []
    for _ in range(cg_steps):
        k_alpha = K_op @ alpha_gd
        grad = k_alpha + lam * alpha_gd - y_s
        alpha_gd = alpha_gd - eta * grad
        gd_errors.append(float(torch.norm(alpha_gd - alpha)))

    f_gd = (K_exp_sq.T @ alpha_gd)

    def rmse(pred: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean((pred - y_q_true) ** 2)))
        
    return {
        "rmse_oracle": rmse(f_oracle),
        "rmse_softmax": rmse(f_softmax),
        "rmse_cg": rmse(f_cg),
        "rmse_gd": rmse(f_gd),
        "rmse_gap": abs(rmse(f_oracle) - rmse(f_softmax)),
        "rmse_cg_gap": abs(rmse(f_oracle) - rmse(f_cg)),
        "rmse_gd_gap": abs(rmse(f_oracle) - rmse(f_gd)),
        "op_norm_diff": float(op_norm_diff),
        "tau": float(tau),
        "lambda": float(lam),
        "cg_steps": cg_steps,
        "cg_errors": cg_errors,
        "gd_errors": gd_errors,
        # Return vectors for plotting
        "f_oracle": f_oracle.tolist(),
        "f_softmax": f_softmax.tolist(),
        "f_cg": f_cg.tolist(),
        "f_gd": f_gd.tolist(),
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
        n_support=int(cfg.get("n_support", 24)),
        n_query=int(cfg.get("n_query", 16)),
        p=int(cfg.get("p", 8)),
        d_proj=int(cfg.get("d_proj", 6)),
        tau=float(cfg.get("tau", 10.0)),
        lam=float(cfg.get("lambda", 1.0)),
        noise=float(cfg.get("noise", 0.1)),
    )
    
    # Print JSON results without the large vectors
    print_res = {k: v for k, v in res.items() if k not in ["f_oracle", "f_softmax", "f_cg", "f_gd", "y_q_true", "cg_errors", "gd_errors"]}
    print(json.dumps(print_res))
    
    if cfg.get("plot", False):
        out_path = cfg.get("out", "docs/figures/route_a_mvp.png")
        try:
            import matplotlib.pyplot as plt
            import os
            
            f_oracle = np.array(res["f_oracle"])
            f_softmax = np.array(res["f_softmax"])
            f_cg = np.array(res["f_cg"])
            f_gd = np.array(res["f_gd"])
            
            # Sort by Oracle value for clarity
            idxs = np.argsort(f_oracle)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left: Predictions comparison
            ax = axes[0]
            ax.plot(f_oracle[idxs], label='Oracle (KRR)', color='blue', marker='o', markersize=4, linewidth=1.5)
            ax.plot(f_cg[idxs], label=f'CG ({res["cg_steps"]} steps)', color='green', marker='s', markersize=3, linestyle='-')
            ax.plot(f_gd[idxs], label=f'GD ({res["cg_steps"]} steps)', color='red', marker='^', markersize=3, linestyle='--')
            ax.plot(f_softmax[idxs], label='1-Layer Kernel Smoother', color='orange', marker='x', markersize=4, linestyle=':', alpha=0.5)
            ax.set_title("Route A: CG vs GD on Softmax Kernel")
            ax.set_xlabel("Query Sample (sorted by Oracle value)")
            ax.set_ylabel("Prediction")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Right: Convergence trajectory (error to oracle vs steps)
            ax2 = axes[1]
            steps_range = list(range(1, len(res["cg_errors"]) + 1))
            ax2.semilogy(steps_range, res["cg_errors"], '-o', color='green', markersize=4, label='CG')
            ax2.semilogy(steps_range, res["gd_errors"], '--^', color='red', markersize=4, label='GD')
            ax2.set_title("Convergence: ||alpha - alpha*||")
            ax2.set_xlabel("Iteration / Layer")
            ax2.set_ylabel("Error (log scale)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot to {out_path}")
        except Exception as e:
            print(f"Error plotting: {e}")


if __name__ == "__main__":
    main()
