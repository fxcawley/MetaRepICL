import torch
import numpy as np
import pytest
from src.softmax.route_b import SoftmaxDotProductApprox

def test_route_b_approximation_accuracy():
    torch.manual_seed(42)
    B, N, D, Dv = 1, 64, 16, 16
    epsilon = 1e-3 # Needs to be small for Taylor expansion to hold
    
    Q = torch.randn(B, N, D)
    K = torch.randn(B, N, D)
    V = torch.randn(B, N, Dv)
    
    # Center V to help approximation (reduces error from the Mean(s)*Sum(v) term)
    V = V - V.mean(dim=1, keepdim=True)
    
    model = SoftmaxDotProductApprox(d_model=D, epsilon=epsilon)
    
    approx = model(Q, K, V)
    exact = model.compute_exact(Q, K, V)
    
    # Relative error
    diff = torch.norm(approx - exact)
    norm = torch.norm(exact)
    rel_err = diff / norm
    
    print(f"Relative Error with epsilon={epsilon}: {rel_err.item()}")
    
    # Expect reasonable approximation for small epsilon
    assert rel_err < 0.05

def test_route_b_scaling():
    # Verify error decreases with epsilon (until precision issues)
    torch.manual_seed(42)
    B, N, D, Dv = 1, 32, 8, 8
    
    Q = torch.randn(B, N, D)
    K = torch.randn(B, N, D)
    V = torch.randn(B, N, Dv)
    
    model_large = SoftmaxDotProductApprox(d_model=D, epsilon=1e-1)
    model_small = SoftmaxDotProductApprox(d_model=D, epsilon=1e-3)
    
    err_large = torch.norm(model_large(Q, K, V) - model_large.compute_exact(Q, K, V))
    err_small = torch.norm(model_small(Q, K, V) - model_small.compute_exact(Q, K, V))
    
    assert err_small < err_large

