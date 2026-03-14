import json
import numpy as np
import hydra
from omegaconf import DictConfig
from src.lat.cg_stack import run_cg

def make_ill_conditioned_phi(n: int, p: int, seed: int, cond: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((p, p)))
    s = np.geomspace(cond, 1.0, p)
    phi_base = rng.standard_normal((n, p))
    phi = phi_base @ (U * s)
    return phi

def run_preconditioned_cg(K, y, lam, t, M_inv_diag):
    """Run preconditioned CG on (K + lam I) alpha = y with diagonal preconditioner.

    M_inv_diag: diagonal entries of M^{-1}, the inverse preconditioner.
    Standard preconditioned CG replaces r-dot-r with r^T M^{-1} r and
    uses z = M^{-1} r as the preconditioned residual.
    """
    n = K.shape[0]
    A = K + lam * np.eye(n)
    alpha = np.zeros(n, dtype=np.float64)
    r = y.copy().astype(np.float64)
    z = M_inv_diag * r  # z = M^{-1} r
    p = z.copy()
    rz = float(r @ z)
    hist = [alpha.copy()]
    for _ in range(int(t)):
        Ap = A @ p
        pAp = float(p @ Ap)
        gamma = rz / (pAp + 1e-18)
        alpha = alpha + gamma * p
        r = r - gamma * Ap
        z = M_inv_diag * r  # z = M^{-1} r
        rz_new = float(r @ z)
        beta = rz_new / (rz + 1e-18)
        p = z + beta * p
        rz = rz_new
        hist.append(alpha.copy())
    return alpha, hist


def run_precond(seed: int = 123, n: int = 128, p: int = 32, lam: float = 1e-2, t: int = 8, cond: float = 1.0):
    rng = np.random.default_rng(seed)
    
    if cond > 1.0:
        phi = make_ill_conditioned_phi(n, p, seed, cond)
    else:
        phi = rng.standard_normal((n, p)).astype(np.float64)
        
    K = phi @ phi.T
    y = rng.standard_normal(n).astype(np.float64)
    alpha_star = np.linalg.solve(K + lam * np.eye(n), y)
    # Unpreconditioned CG
    errs_un = []
    for step in range(1, t + 1):
        alpha, r, pvec = run_cg(phi, y, lam, step)
        errs_un.append(float(np.linalg.norm(alpha - alpha_star)))
    # Jacobi-preconditioned CG: M = diag(K + lam I), M^{-1} = 1/diag(A)
    diagA = np.diag(K) + lam
    M_inv_diag = 1.0 / (diagA + 1e-6)
    errs_pr = []
    for step in range(1, t + 1):
        alpha_pr, _ = run_preconditioned_cg(K, y, lam, step, M_inv_diag)
        errs_pr.append(float(np.linalg.norm(alpha_pr - alpha_star)))
    return {"err_un": errs_un, "err_pr": errs_pr}


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    res = run_precond(
        seed=int(cfg.get("seed", 123)),
        n=int(cfg.get("n_support", 128)),
        p=int(cfg.get("p", 32)),
        lam=float(cfg.get("lambda", 1e-2)),
        t=int(cfg.get("steps", 8)),
        cond=float(cfg.get("cond", 1.0))
    )
    print(json.dumps(res))


if __name__ == "__main__":
	main()
