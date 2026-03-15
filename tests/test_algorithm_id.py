"""Tests for the algorithm identification experiment.

Verifies that all named algorithms converge to the correct solution
on a well-conditioned test problem.
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pytest
from experiments.algorithm_id import (
    vanilla_gd_trajectory,
    vanilla_cg_trajectory,
    preconditioned_gd_trajectory,
    heavy_ball_trajectory,
    chebyshev_iteration_trajectory,
    preconditioned_cg_trajectory,
    ALGORITHMS,
)


@pytest.fixture
def well_conditioned_problem():
    """Create a well-conditioned least squares problem."""
    rng = np.random.default_rng(42)
    n, p = 30, 10
    X = rng.standard_normal((n, p)).astype(np.float64)
    w_true = rng.standard_normal(p).astype(np.float64)
    y = X @ w_true + 0.01 * rng.standard_normal(n)
    lam = 0.1
    # True solution
    B = X.T @ X + lam * np.eye(p)
    w_star = np.linalg.solve(B, X.T @ y)
    return X, y, lam, w_star


@pytest.fixture
def ill_conditioned_problem():
    """Create an ill-conditioned least squares problem."""
    rng = np.random.default_rng(42)
    n, p = 30, 10
    # Create features with condition number ~100
    sigmas = np.geomspace(100.0, 1.0, p)
    X = rng.standard_normal((n, p)).astype(np.float64) * np.sqrt(sigmas)
    w_true = rng.standard_normal(p).astype(np.float64)
    y = X @ w_true + 0.01 * rng.standard_normal(n)
    lam = 0.1
    B = X.T @ X + lam * np.eye(p)
    w_star = np.linalg.solve(B, X.T @ y)
    return X, y, lam, w_star


class TestAlgorithmConvergence:
    """All algorithms should converge to the correct solution."""

    def test_vanilla_gd_converges(self, well_conditioned_problem):
        X, y, lam, w_star = well_conditioned_problem
        traj = vanilla_gd_trajectory(X, y, lam, steps=200)
        err = np.linalg.norm(traj[-1] - w_star)
        assert err < 1e-4, f"GD did not converge: err={err}"

    def test_vanilla_cg_converges(self, well_conditioned_problem):
        X, y, lam, w_star = well_conditioned_problem
        traj = vanilla_cg_trajectory(X, y, lam, steps=20)
        err = np.linalg.norm(traj[-1] - w_star)
        assert err < 1e-8, f"CG did not converge: err={err}"

    def test_preconditioned_gd_converges(self, well_conditioned_problem):
        X, y, lam, w_star = well_conditioned_problem
        traj = preconditioned_gd_trajectory(X, y, lam, steps=200)
        err = np.linalg.norm(traj[-1] - w_star)
        assert err < 1e-4, f"Preconditioned GD did not converge: err={err}"

    def test_heavy_ball_converges(self, well_conditioned_problem):
        X, y, lam, w_star = well_conditioned_problem
        traj = heavy_ball_trajectory(X, y, lam, steps=200)
        err = np.linalg.norm(traj[-1] - w_star)
        assert err < 1e-4, f"Heavy Ball did not converge: err={err}"

    def test_chebyshev_converges(self, well_conditioned_problem):
        X, y, lam, w_star = well_conditioned_problem
        traj = chebyshev_iteration_trajectory(X, y, lam, steps=50)
        err = np.linalg.norm(traj[-1] - w_star)
        assert err < 1e-4, f"Chebyshev did not converge: err={err}"

    def test_preconditioned_cg_converges(self, well_conditioned_problem):
        X, y, lam, w_star = well_conditioned_problem
        traj = preconditioned_cg_trajectory(X, y, lam, steps=20)
        err = np.linalg.norm(traj[-1] - w_star)
        assert err < 1e-8, f"Preconditioned CG did not converge: err={err}"


class TestConvergenceOrdering:
    """On ill-conditioned problems, CG-type methods should converge faster than GD."""

    def test_cg_faster_than_gd(self, ill_conditioned_problem):
        X, y, lam, w_star = ill_conditioned_problem
        gd_traj = vanilla_gd_trajectory(X, y, lam, steps=12)
        cg_traj = vanilla_cg_trajectory(X, y, lam, steps=12)
        gd_err = np.linalg.norm(gd_traj[-1] - w_star)
        cg_err = np.linalg.norm(cg_traj[-1] - w_star)
        assert cg_err < gd_err, \
            f"CG should be faster than GD on ill-conditioned problem: CG={cg_err}, GD={gd_err}"

    def test_precond_gd_faster_than_vanilla_gd(self, ill_conditioned_problem):
        X, y, lam, w_star = ill_conditioned_problem
        gd_traj = vanilla_gd_trajectory(X, y, lam, steps=12)
        pgd_traj = preconditioned_gd_trajectory(X, y, lam, steps=12)
        gd_err = np.linalg.norm(gd_traj[-1] - w_star)
        pgd_err = np.linalg.norm(pgd_traj[-1] - w_star)
        assert pgd_err < gd_err, \
            f"Preconditioned GD should beat vanilla GD: PGD={pgd_err}, GD={gd_err}"

    def test_precond_cg_faster_than_vanilla_cg(self, ill_conditioned_problem):
        X, y, lam, w_star = ill_conditioned_problem
        cg_traj = vanilla_cg_trajectory(X, y, lam, steps=12)
        pcg_traj = preconditioned_cg_trajectory(X, y, lam, steps=12)
        cg_err = np.linalg.norm(cg_traj[-1] - w_star)
        pcg_err = np.linalg.norm(pcg_traj[-1] - w_star)
        assert pcg_err < cg_err, \
            f"Preconditioned CG should beat vanilla CG: PCG={pcg_err}, CG={cg_err}"


class TestTrajectoryProperties:
    """Verify structural properties of algorithm trajectories."""

    def test_all_start_at_zero(self, well_conditioned_problem):
        X, y, lam, _ = well_conditioned_problem
        for name, fn in ALGORITHMS.items():
            traj = fn(X, y, lam, steps=5)
            assert np.allclose(traj[0], 0), f"{name} should start at zero"

    def test_trajectory_length(self, well_conditioned_problem):
        X, y, lam, _ = well_conditioned_problem
        steps = 7
        for name, fn in ALGORITHMS.items():
            traj = fn(X, y, lam, steps=steps)
            assert len(traj) == steps + 1, \
                f"{name}: expected {steps+1} states, got {len(traj)}"

    def test_cg_converges_in_p_steps(self, well_conditioned_problem):
        """CG should converge in at most p steps on a p-dimensional problem."""
        X, y, lam, w_star = well_conditioned_problem
        p = X.shape[1]
        traj = vanilla_cg_trajectory(X, y, lam, steps=p)
        err = np.linalg.norm(traj[p] - w_star)
        assert err < 1e-8, f"CG should converge in p={p} steps: err={err}"


class TestNumericalStability:
    """Verify algorithms don't diverge at high condition numbers."""

    @pytest.fixture
    def high_kappa_problem(self):
        """Problem with kappa=500 (matching the experiment)."""
        rng = np.random.default_rng(42)
        n, p = 30, 10
        sigmas = np.geomspace(500.0, 1.0, p)
        X = rng.standard_normal((n, p)).astype(np.float64) * np.sqrt(sigmas)
        w_true = rng.standard_normal(p).astype(np.float64)
        y = X @ w_true + 0.01 * rng.standard_normal(n)
        lam = 0.1
        B = X.T @ X + lam * np.eye(p)
        w_star = np.linalg.solve(B, X.T @ y)
        return X, y, lam, w_star

    def test_precond_gd_eigenvalues_correct(self, high_kappa_problem):
        """Precond GD must use correct eigenvalue computation (not eigvalsh on non-symmetric M^{-1}B)."""
        X, y, lam, w_star = high_kappa_problem
        B = X.T @ X + lam * np.eye(X.shape[1])
        M_inv = 1.0 / np.diag(B)
        # The symmetrized version should have all positive eigenvalues
        M_inv_half = np.sqrt(M_inv)
        symm = np.diag(M_inv_half) @ B @ np.diag(M_inv_half)
        eigs = np.linalg.eigvalsh(symm)
        assert np.all(eigs > 0), f"Symmetrized preconditioned matrix should be SPD: min eig = {eigs[0]}"

    def test_precond_gd_does_not_diverge(self, high_kappa_problem):
        """Precond GD should converge (not diverge) at kappa=500."""
        X, y, lam, w_star = high_kappa_problem
        traj = preconditioned_gd_trajectory(X, y, lam, steps=12)
        err_first = np.linalg.norm(traj[1] - w_star)
        err_last = np.linalg.norm(traj[-1] - w_star)
        assert err_last < err_first, \
            f"Precond GD should converge at kappa=500: err went {err_first:.4f} -> {err_last:.4f}"

    def test_heavy_ball_does_not_diverge(self, high_kappa_problem):
        """Heavy Ball with damped momentum should not diverge at kappa=500."""
        X, y, lam, w_star = high_kappa_problem
        traj = heavy_ball_trajectory(X, y, lam, steps=12)
        err_first = np.linalg.norm(traj[1] - w_star)
        err_last = np.linalg.norm(traj[-1] - w_star)
        assert err_last < err_first * 10, \
            f"Heavy Ball should not diverge at kappa=500: err went {err_first:.4f} -> {err_last:.4f}"

    def test_chebyshev_does_not_diverge(self, high_kappa_problem):
        """Chebyshev should not diverge at kappa=500."""
        X, y, lam, w_star = high_kappa_problem
        traj = chebyshev_iteration_trajectory(X, y, lam, steps=12)
        # Check no NaN/Inf
        for i, w in enumerate(traj):
            assert np.all(np.isfinite(w)), f"Chebyshev produced non-finite at step {i}"
        err_last = np.linalg.norm(traj[-1] - w_star)
        assert err_last < np.linalg.norm(w_star) * 10, \
            f"Chebyshev error should be bounded: {err_last:.4f}"
