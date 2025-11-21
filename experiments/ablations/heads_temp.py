import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import argparse
import os
import sys
from pathlib import Path

# Add src to path if needed
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from src.softmax.route_b import SoftmaxDotProductApprox
except ImportError:
    pass

# Patch argparse for Hydra/Py3.14
_orig_add_argument = argparse.ArgumentParser.add_argument
def _safe_add_argument(self, *args, **kwargs):
    if 'help' in kwargs:
        h = kwargs['help']
        if hasattr(h, '__class__') and h.__class__.__name__ == 'LazyCompletionHelp':
            kwargs['help'] = "Shell completion"
    return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _safe_add_argument

def run_route_b_ablation(seed=123, n=64, p=16, eps_list=None):
    """
    Sweep epsilon for Route B approximation.
    Compare:
    1. Full (Head1 - Head2)
    2. Ablated (Head1 only)
    """
    if eps_list is None:
        eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    torch.manual_seed(seed)
    Q = torch.randn(1, n, p)
    K = torch.randn(1, n, p)
    V = torch.randn(1, n, 1) # Use scalar values for simplicity
    
    # True unnormalized dot product attention
    exact = torch.bmm(torch.bmm(Q, K.transpose(1, 2)), V)
    norm_exact = torch.norm(exact)
    
    errs_full = []
    errs_ablated = []
    
    for eps in eps_list:
        model = SoftmaxDotProductApprox(d_model=p, epsilon=eps)
        
        # Manual forward to get components
        scale = eps
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale
        attn_probs = torch.softmax(scores, dim=-1)
        head1 = torch.bmm(attn_probs, V)
        
        mean_v = V.mean(dim=1, keepdim=True)
        head2 = mean_v.expand_as(head1)
        
        # Full: (H1 - H2) * (n/eps)
        approx_full = (head1 - head2) * (n / eps)
        
        # Ablated: H1 * (n/eps) -> this is just scaled softmax
        # Usually this doesn't approximate dot product well unless V is mean-zero and centered?
        approx_ablated = head1 * (n / eps)
        
        err_f = torch.norm(approx_full - exact) / norm_exact
        err_a = torch.norm(approx_ablated - exact) / norm_exact
        
        errs_full.append(float(err_f))
        errs_ablated.append(float(err_a))
        
    return {
        "eps": eps_list,
        "err_full": errs_full,
        "err_ablated": errs_ablated
    }

def run_route_a_ablation(seed=123, n=48, p=16, tau_list=None, lam=1e-2):
    """
    Sweep temperature for Route A (Softmax vs Exp Kernel).
    Compare RMSE of:
    1. Oracle KRR with Exp Kernel K(x,y) = exp(<x,y>/tau)
    2. Softmax Weighted Regression (Nadaraya-Watson) f(x) = sum w_i y_i
       where w_i = softmax(<x, x_i>/tau)
       
    Hypothesis: As tau -> 0, softmax approaches 1-NN?
    As tau -> inf, softmax approaches mean.
    Exp kernel behavior depends on tau too.
    """
    if tau_list is None:
        tau_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    
    # Data
    phi_s = torch.randn(n, p).double()
    phi_q = torch.randn(1, p).double()
    
    # True function (linear)
    w_true = torch.randn(p, 1).double()
    y_s = phi_s @ w_true + 0.1 * torch.randn(n, 1).double()
    y_q_true = phi_q @ w_true
    
    rmse_oracle = []
    rmse_softmax = []
    
    for tau in tau_list:
        # Kernel matrices
        K_ss = torch.exp( (phi_s @ phi_s.T) / tau )
        K_sq = torch.exp( (phi_s @ phi_q.T) / tau ) # (n, 1)
        
        # Oracle KRR
        # (K + lam I) alpha = y
        A = K_ss + lam * torch.eye(n).double()
        try:
            alpha = torch.linalg.solve(A, y_s)
            f_oracle = K_sq.T @ alpha
            err_o = float(torch.abs(f_oracle - y_q_true))
        except:
            err_o = float('nan')
            
        # Softmax Baseline (Nadaraya-Watson)
        # Weights = softmax( <q, s> / tau )
        # Note: softmax is over the support dimension
        # scores = (phi_q @ phi_s.T) / tau -> (1, n)
        scores = (phi_q @ phi_s.T) / tau
        weights = torch.softmax(scores, dim=1).T # (n, 1)
        f_soft = weights.T @ y_s
        err_s = float(torch.abs(f_soft - y_q_true))
        
        rmse_oracle.append(err_o)
        rmse_softmax.append(err_s)
        
    return {
        "tau": tau_list,
        "rmse_oracle": rmse_oracle,
        "rmse_softmax": rmse_softmax
    }

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed = int(cfg.get("seed", 123))
    
    # 1. Route B Ablation (Epsilon Sweep)
    print("Running Route B Ablation (Head Sharing / Epsilon)...")
    eps_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]
    res_b = run_route_b_ablation(seed=seed, eps_list=eps_list)
    
    # 2. Route A Ablation (Temperature/Normalization)
    print("Running Route A Ablation (Temperature/Normalization)...")
    tau_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    res_a = run_route_a_ablation(seed=seed, tau_list=tau_list)
    
    # Plotting
    if cfg.get("plot", False) or True: # Always plot for now, or controlled by cfg
        os.makedirs("figures/ablations", exist_ok=True)
        
        # Plot B
        plt.figure(figsize=(10, 5))
        plt.loglog(res_b["eps"], res_b["err_full"], '-o', label="Full (2-Head)")
        plt.loglog(res_b["eps"], res_b["err_ablated"], '--x', label="Ablated (1-Head)")
        plt.xlabel("Epsilon")
        plt.ylabel("Rel Error to Linear Attn")
        plt.title("Route B: Head Sharing vs Dedicated Aggregation Head")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("figures/ablations/route_b_heads.png")
        print("Saved figures/ablations/route_b_heads.png")
        
        # Plot A
        plt.figure(figsize=(10, 5))
        plt.semilogx(res_a["tau"], res_a["rmse_oracle"], '-o', label="Oracle KRR (Exp Kernel)")
        plt.semilogx(res_a["tau"], res_a["rmse_softmax"], '--x', label="Softmax Regression (Normalized)")
        plt.xlabel("Temperature (tau)")
        plt.ylabel("Prediction Error (Abs)")
        plt.title("Route A: Normalization Effects (Exp vs Softmax)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("figures/ablations/route_a_temp.png")
        print("Saved figures/ablations/route_a_temp.png")

if __name__ == "__main__":
    main()

