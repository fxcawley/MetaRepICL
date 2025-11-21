import numpy as np
import pytest
from src.data.glm import make_glm_dataset

def test_glm_shapes():
    n_s, n_q, p = 32, 16, 8
    Xs, ys, Xq, yq = make_glm_dataset(n_s, n_q, p, 'logistic', seed=42)
    
    assert Xs.shape == (n_s, p)
    assert ys.shape == (n_s,)
    assert Xq.shape == (n_q, p)
    assert yq.shape == (n_q,)

def test_glm_logistic_values():
    Xs, ys, Xq, yq = make_glm_dataset(32, 16, 8, 'logistic', seed=42)
    # Logistic labels should be 0 or 1
    unique = np.unique(ys)
    for u in unique:
        assert u in [0.0, 1.0]

def test_glm_poisson_values():
    Xs, ys, Xq, yq = make_glm_dataset(32, 16, 8, 'poisson', seed=42)
    # Poisson labels should be non-negative integers
    assert np.all(ys >= 0)
    # Check if they are roughly integers (stored as float)
    assert np.all(np.mod(ys, 1) == 0)

def test_glm_determinism():
    Xs1, ys1, _, _ = make_glm_dataset(32, 16, 8, 'logistic', seed=123)
    Xs2, ys2, _, _ = make_glm_dataset(32, 16, 8, 'logistic', seed=123)
    Xs3, ys3, _, _ = make_glm_dataset(32, 16, 8, 'logistic', seed=124)
    
    np.testing.assert_array_equal(Xs1, Xs2)
    np.testing.assert_array_equal(ys1, ys2)
    assert not np.array_equal(ys1, ys3)

