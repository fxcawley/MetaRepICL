import os
import numpy as np
import matplotlib.pyplot as plt
# import hydra
# from omegaconf import DictConfig
from src.lat.cg_stack import run_cg
from src.lat.preconditioner import build_diag_preconditioner

def make_ill_conditioned_phi(n: int, p: int, seed: int, cond: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Create random orthogonal basis
    U, _ = np.linalg.qr(rng.standard_normal((p, p)))
    # Set singular values to achieve condition number
    s = np.geomspace(cond, 1.0, p)
    # Create phi with these singular values
    # We want K = phi @ phi.T to have high condition number.
    # K's eigenvalues are s^2 (if n >= p).
    # So condition number of K will be cond^2.
    # Wait, the user usually specifies condition number of the OPERATOR K.
    # If input cond is for phi, K has cond^2. 
    # Let's assume 'cond' is the target condition number of K's spectrum (approx).
    # So we set singular values of phi to sqrt(decay).
    s = np.sqrt(np.geomspace(cond, 1.0, p))
    
    # Embed in n dimensions (n > p usually for these experiments)
    # Random projection from p to n?
    # Or just first p dimensions?
    # Let's make phi (n x p) so K is (n x n) rank p.
    # For ICL, typically we have support size n, dimension d (here p).
    phi_base = rng.standard_normal((n, p))
    # Force covariance: phi = Z @ (U * s) where Z is whitenened?
    # Actually, simpler: just create (p,p) covariance and sample.
    # Or simpler: Phi = U_n @ S @ V_p.T
    # Let's stick to the precond.py implementation but fixing the cond interpretation.
    # In precond.py: s = geomspace(cond, 1.0, p); phi = phi_base @ (U*s).
    # This effectively multiplies directions by s. 
    phi = phi_base @ (U * s)
    return phi

def solve_exact(phi, y, lam):
    n = phi.shape[0]
    K = phi @ phi.T
    return np.linalg.solve(K + lam * np.eye(n), y)

def run_experiment(n, p, seed, cond, lam, steps):
    phi = make_ill_conditioned_phi(n, p, seed, cond)
    rng = np.random.default_rng(seed + 1)
    y = rng.standard_normal(n)
    
    # Exact solution for reference
    alpha_star = solve_exact(phi, y, lam)
    
    # 1. Standard CG
    # We want to track error per step. run_cg returns final, but we need history.
    # cg_stack.py has run_cg_with_history?
    # Let's check cg_stack.py again. It does!
    from src.lat.cg_stack import run_cg_with_history
    
    hist_std = run_cg_with_history(phi, y, lam, steps)
    errs_std = [np.linalg.norm(h[0] - alpha_star) for h in hist_std]
    # Prepend initial error (0 vector)
    errs_std.insert(0, np.linalg.norm(alpha_star))
    
    # 2. Preconditioned CG (via transformation)
    # P^{-1} = 1/(diag(K) + lam)
    K = phi @ phi.T
    diagK = np.diag(K)
    pinv = build_diag_preconditioner(diagK + lam) # returns 1/(d+eps)
    # S = P^{-1/2} = sqrt(pinv)
    S = np.sqrt(pinv)
    
    # Transform problem:
    # phi_tilde = S[:, None] * phi
    # y_tilde = S * y
    phi_tilde = S[:, None] * phi
    y_tilde = S * y
    
    # Effective lambda?
    # We are solving (S K S + lam I) alpha_tilde = S y
    # Equivalent to (K + lam S^{-2}) alpha = y where alpha = S alpha_tilde
    # We want to compare to the solution of THIS problem or the original?
    # The failure mode demo should ideally show we solve the ORIGINAL problem better?
    # Or that we solve the MODIFIED problem fast, and the modified problem is "good enough".
    # Usually P is an approximation of (K+lam I).
    # S^{-2} = diag(K) + lam.
    # So we are solving (K + lam * diag(K) + lam^2) alpha = y?
    # No, (K + lam * (diag(K)+lam)) alpha ... wait.
    # (S K S + lam I) -> S (K + lam S^{-2}) S.
    # The effective regularizer is lam * S^{-2} = lam * (diag(K) + lam).
    # This is a DIFFERENT problem.
    # However, let's just compute convergence to the solution of THIS transformed problem
    # and see if it's faster than Standard CG on the original problem (normalized?).
    
    # Actually, standard PCG targets the original solution.
    # If we use the transform method, we effectively change the regularizer.
    # Let's track convergence to the exact solution of the TRANSFORMED problem 
    # (mapped back to original space) to show optimizer speed.
    
    alpha_star_pre = solve_exact(phi_tilde, y_tilde, lam)
    # Map back: alpha = S * alpha_tilde
    target_alpha_pre_orig = S * alpha_star_pre
    
    hist_pre = run_cg_with_history(phi_tilde, y_tilde, lam, steps)
    errs_pre = []
    for (a_tilde, _, _) in hist_pre:
        a_orig = S * a_tilde
        errs_pre.append(np.linalg.norm(a_orig - target_alpha_pre_orig))
    errs_pre.insert(0, np.linalg.norm(target_alpha_pre_orig))
    
    return errs_std, errs_pre

def main():
    # Parameters
    n = 128
    p = 64
    seed = 42
    lam = 0.01
    steps = 15
    
    # Sweep condition numbers
    conds = [1.0, 10.0, 100.0, 1000.0]
    
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(conds)))
    
    for i, cond in enumerate(conds):
        print(f"Running for condition number: {cond}")
        errs_std, errs_pre = run_experiment(n, p, seed, cond, lam, steps)
        
        # Plot
        plt.plot(errs_std, linestyle='-', color=colors[i], label=f'Std (cond={cond})')
        plt.plot(errs_pre, linestyle='--', color=colors[i], label=f'Pre (cond={cond})')
        
        results[cond] = {
            "std": errs_std[-1],
            "pre": errs_pre[-1]
        }
        
    plt.yscale('log')
    plt.xlabel('CG Steps')
    plt.ylabel('Error ||alpha - alpha*||')
    plt.title(f'CG Stall & Recovery (n={n}, p={p}, lam={lam})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("figures/failure_modes", exist_ok=True)
    out_path = "figures/failure_modes/ill_conditioned_cg.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()

