import numpy as np
from typing import Tuple, Literal

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def make_glm_dataset(
    n_support: int,
    n_query: int,
    p: int,
    family: Literal['logistic', 'poisson'],
    noise: float = 0.0,
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for Generalized Linear Models.
    
    Args:
        n_support: Number of support examples.
        n_query: Number of query examples.
        p: Feature dimension.
        family: 'logistic' or 'poisson'.
        noise: Additive noise to the linear predictor before link (optional) or label noise?
               Standard GLM noise comes from the sampling distribution.
               Here 'noise' parameter might be used for extra variance in x or w.
               We'll use it to perturb the linear predictor: z = x^T w + noise * eps.
        seed: Random seed.
        
    Returns:
        (Xs, ys, Xq, yq)
    """
    rng = np.random.default_rng(seed)
    n_total = n_support + n_query
    
    # Generate features X ~ N(0, I)
    X = rng.standard_normal((n_total, p)).astype(np.float64)
    
    # Generate true weights w ~ N(0, I/p)
    w = rng.standard_normal(p).astype(np.float64) / np.sqrt(p)
    
    # Linear predictor
    z = X @ w + noise * rng.standard_normal(n_total)
    
    if family == 'logistic':
        # p = sigmoid(z)
        probs = sigmoid(z)
        # Sample y ~ Bernoulli(p)
        y = rng.binomial(1, probs).astype(np.float64)
    elif family == 'poisson':
        # lambda = exp(z)
        # We need to be careful with exploding gradients/values with exp(z) if z is large.
        # w is scaled by 1/sqrt(p), X is standard normal -> z ~ N(0, 1).
        # exp(N(0,1)) is reasonable.
        rate = np.exp(z)
        y = rng.poisson(rate).astype(np.float64)
    else:
        raise ValueError(f"Unknown family: {family}")
        
    Xs, Xq = X[:n_support], X[n_support:]
    ys, yq = y[:n_support], y[n_support:]
    
    return Xs, ys, Xq, yq

