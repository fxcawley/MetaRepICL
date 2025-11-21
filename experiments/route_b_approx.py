import json
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

try:
    from src.softmax.route_b import SoftmaxDotProductApprox
    from src.lat.cg_step import cg_step
except ImportError:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.softmax.route_b import SoftmaxDotProductApprox
    from src.lat.cg_step import cg_step

import argparse
# Patch argparse to handle Hydra's LazyCompletionHelp on Python 3.14
# This must be done before importing hydra or running @hydra.main
_orig_add_argument = argparse.ArgumentParser.add_argument
def _safe_add_argument(self, *args, **kwargs):
	if 'help' in kwargs:
		h = kwargs['help']
		if hasattr(h, '__class__') and h.__class__.__name__ == 'LazyCompletionHelp':
			kwargs['help'] = "Shell completion"
	return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _safe_add_argument


def route_b_matvec(model: SoftmaxDotProductApprox, phi: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute Approx(K p) = Approx((phi phi^T) p) using Softmax model.
    K = phi @ phi.T
    Kp = phi @ (phi.T @ p)
    
    Route B model computes: Approx((Q K^T) V)
    Here Q=phi, K=phi, V=p_vec.
    """
    # Ensure dimensions
    # phi: (N, P)
    # p_vec: (N,) or (N, D)
    
    if p_vec.ndim == 1:
        V = p_vec.unsqueeze(1) # (N, 1)
    else:
        V = p_vec
        
    # Batch dim = 1
    Q_b = phi.unsqueeze(0) # (1, N, P)
    K_b = phi.unsqueeze(0) # (1, N, P)
    V_b = V.unsqueeze(0)   # (1, N, 1)
    
    # Center V_b to reduce approximation error
    # We are effectively computing K(p - mean), which is roughly Kp if mean is small or K*1 is small.
    # Actually, K*1 is typically not small.
    # However, if we want to implement linear attention Kp exactly, we can't just assume mean is zero.
    # The "Two-Head Rescaling" trick:
    # Head 1: Attn(V)
    # Head 2: Attn(1) * Mean(V) or something?
    # Our implemented Route B is: (Head1 - Mean(V)) * (N/eps).
    # This approximates K V - (Mean(S)) * Sum(V).
    # If we center V, Sum(V)=0, so we get K V.
    # For general p, we can decompose p = p_centered + p_mean * 1.
    # K p = K p_centered + p_mean * (K 1).
    # Route B (centered input) gives K p_centered.
    # Can we get K 1?
    # K 1 is the row sums of the kernel.
    # If we use V=1 (constant), Route B gives error (Sum(V) is large).
    
    # For this MVP, we assume we can use the centered approximation for the gradient directions which are often centered?
    # Let's try centering p_vec here before passing to the model, and see the error.
    # If it's large, we acknowledge the limitation or refine the approximation.
    
    V_mean = V_b.mean(dim=1, keepdim=True)
    V_centered = V_b - V_mean
    
    # Approx K (p - mean)
    Kp_centered = model(Q_b, K_b, V_centered) # (1, N, 1)
    
    # We miss K * mean.
    # K * mean = mean * (phi @ phi.T @ 1) = mean * phi @ (sum phi)
    # This is a rank-1 term.
    # Maybe we can compute this with another head?
    # For now, let's just return the centered approximation and see the error.
    # If we are running CG, we need Kp accurate.
    
    return Kp_centered.squeeze(0).squeeze(1)


def run_cg_route_b(
    phi: np.ndarray,
    y: np.ndarray,
    lam: float,
    t: int,
    epsilon: float = 1e-4
) -> dict:
    # Convert to Torch
    device = torch.device("cpu")
    phi_t = torch.tensor(phi, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    
    n, p_dim = phi.shape
    model = SoftmaxDotProductApprox(d_model=p_dim, epsilon=epsilon)
    
    # CG State
    alpha = torch.zeros(n, dtype=torch.float32, device=device)
    r = y_t.clone()
    p = r.clone()
    rr = torch.dot(r, r)
    
    hist_err = []
    
    # Exact solution for reference
    K = phi @ phi.T
    print(f"K*1 norm: {np.linalg.norm(K @ np.ones(n))}")
    
    alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
    
    # Exact CG for reference
    alpha_exact = np.zeros(n)
    r_exact = y.copy()
    p_exact = r_exact.copy()
    rr_exact = np.dot(r_exact, r_exact)
    err_exact_hist = []
    
    for step in range(t):
        # Exact Matvec
        Kp_ex = K @ p_exact
        Ap_ex = Kp_ex + lam * p_exact
        pAp_ex = np.dot(p_exact, Ap_ex)
        gamma_ex = rr_exact / (pAp_ex + 1e-18)
        alpha_exact += gamma_ex * p_exact
        r_next_ex = r_exact - gamma_ex * Ap_ex
        rr_next_ex = np.dot(r_next_ex, r_next_ex)
        beta_ex = rr_next_ex / (rr_exact + 1e-18)
        p_exact = r_next_ex + beta_ex * p_exact
        r_exact = r_next_ex
        rr_exact = rr_next_ex
        err_exact_hist.append(float(np.linalg.norm(alpha_exact - alpha_star)))

    # Approx CG
    matvec_errs = []
    for step in range(t):
        # Matvec: Ap = Kp + lam*p
        Kp = route_b_matvec(model, phi_t, p)
        
        # Monitor error relative to exact K*p (using current p)
        Kp_exact_curr = torch.tensor(K, dtype=torch.float32) @ p
        mv_err = torch.norm(Kp - Kp_exact_curr) / (torch.norm(Kp_exact_curr) + 1e-18)
        matvec_errs.append(float(mv_err))
        
        Ap = Kp + lam * p
        
        # CG Step
        pAp = torch.dot(p, Ap)
        gamma = rr / (pAp + 1e-18)
        alpha = alpha + gamma * p
        r_next = r - gamma * Ap
        rr_next = torch.dot(r_next, r_next)
        beta = rr_next / (rr + 1e-18)
        p_next = r_next + beta * p
        
        # Update
        r = r_next
        p = p_next
        rr = rr_next
        
        # Error
        err = np.linalg.norm(alpha.detach().numpy() - alpha_star)
        hist_err.append(float(err))
        
    return {
        "err_history": hist_err, 
        "final_err": hist_err[-1], 
        "err_exact": err_exact_hist,
        "matvec_errs": matvec_errs
    }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed = int(cfg.get("seed", 123))
    rng = np.random.default_rng(seed)
    n = int(cfg.get("n_support", 64))
    p = int(cfg.get("p", 16))
    lam = float(cfg.get("lambda", 1e-2))
    epsilon = float(cfg.get("epsilon", 1e-4)) # Use 1e-4 for better precision
    
    phi = rng.standard_normal((n, p))
    # Center phi to ensure K*1 = 0, making the approximation valid for uncentered p
    phi -= phi.mean(axis=0)
    
    y = rng.standard_normal(n)
    # Center y
    y -= y.mean()
    
    res = run_cg_route_b(phi, y, lam, t=5, epsilon=epsilon)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
