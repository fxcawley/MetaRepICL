"""Float32 vs float64 parity tests.

Verifies that float32 computations produce results within 1% of float64
reference values on fixed seeds, as required by the numerics policy
(REPRODUCIBILITY.md).
"""

import numpy as np
import pytest

from src.lat.matvec import attention_matvec
from src.lat.cg_stack import run_cg
from src.data.synth_linear import make_linear_dataset, krr_oracle


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error between a and b, with b as reference."""
    denom = np.linalg.norm(b)
    if denom < 1e-12:
        return np.linalg.norm(a - b)
    return float(np.linalg.norm(a - b) / denom)


class TestMatvecParity:
    """attention_matvec should agree within 1% across dtypes."""

    def test_matvec_float32_vs_float64(self):
        rng = np.random.default_rng(42)
        n, p = 32, 8
        phi_64 = rng.standard_normal((n, p)).astype(np.float64)
        v_64 = rng.standard_normal(n).astype(np.float64)

        phi_32 = phi_64.astype(np.float32)
        v_32 = v_64.astype(np.float32)

        result_64 = attention_matvec(phi_64, phi_64, v_64)
        result_32 = attention_matvec(phi_32, phi_32, v_32)

        err = _relative_error(result_32.astype(np.float64), result_64)
        assert err < 0.01, f"Matvec parity error {err:.6f} exceeds 1%"


class TestCGStackParity:
    """CG solver should agree within 1% across dtypes.

    Note: run_cg currently promotes inputs to float64 internally, so this
    test validates that the public API behaves consistently regardless of
    the caller's input dtype.  If the internal promotion is ever removed,
    these tests become the safety net that catches precision regressions.
    """

    @pytest.mark.parametrize("t", [1, 3, 6])
    def test_cg_parity(self, t):
        rng = np.random.default_rng(123)
        n, p = 16, 8
        phi_64 = rng.standard_normal((n, p)).astype(np.float64)
        y_64 = rng.standard_normal(n).astype(np.float64)

        phi_32 = phi_64.astype(np.float32)
        y_32 = y_64.astype(np.float32)

        alpha_64, _, _ = run_cg(phi_64, y_64, lam=1e-1, t=t)
        alpha_32, _, _ = run_cg(phi_32, y_32, lam=1e-1, t=t)

        err = _relative_error(alpha_32.astype(np.float64), alpha_64)
        assert err < 0.01, f"CG parity error at t={t}: {err:.6f} exceeds 1%"


class TestKRRParity:
    """KRR oracle should agree within 1% across dtypes."""

    def test_krr_oracle_parity(self):
        # make_linear_dataset returns (X_support, y_support, X_query, y_query)
        X_s, y_s, X_q, _y_q = make_linear_dataset(
            n_support=32, n_query=16, p=8, noise=0.1, seed=77,
        )

        X_s64 = X_s.astype(np.float64)
        y_s64 = y_s.astype(np.float64)
        X_q64 = X_q.astype(np.float64)

        X_s32 = X_s64.astype(np.float32)
        y_s32 = y_s64.astype(np.float32)
        X_q32 = X_q64.astype(np.float32)

        pred_64 = krr_oracle(X_s64, y_s64, X_q64, lam=1e-2)
        pred_32 = krr_oracle(X_s32, y_s32, X_q32, lam=1e-2)

        err = _relative_error(pred_32.astype(np.float64), pred_64)
        assert err < 0.01, f"KRR oracle parity error {err:.6f} exceeds 1%"
