import json
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import os

try:
    from src.softmax.route_b import SoftmaxDotProductApprox
except ImportError:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.softmax.route_b import SoftmaxDotProductApprox

import argparse
# Patch argparse for Hydra/Py3.14
_orig_add_argument = argparse.ArgumentParser.add_argument
def _safe_add_argument(self, *args, **kwargs):
    if 'help' in kwargs:
        h = kwargs['help']
        if hasattr(h, '__class__') and h.__class__.__name__ == 'LazyCompletionHelp':
            kwargs['help'] = "Shell completion"
    return _orig_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = _safe_add_argument


def run_head_drop_ablation(
    seed: int = 123,
    n: int = 64,
    p: int = 16,
    epsilon: float = 1e-4
):
    """
    Route B implements Kp approx using two heads:
    Head 1: Scaled Softmax Attention (approximates Kp + terms)
    Head 2: Uniform Attention (approximates Mean correction)
    
    We ablate Head 2 to see the impact of 'missing the aggregation head'.
    """
    torch.manual_seed(seed)
    
    # Setup single matvec instance
    # Random Q, K, V
    # Use UNCENTERED V to demonstrate the necessity of the second head (aggregator)
    
    Q = torch.randn(1, n, p)
    K = torch.randn(1, n, p)
    V = torch.randn(1, n, 1)
    
    model = SoftmaxDotProductApprox(d_model=p, epsilon=epsilon)
    
    # Full Model (Head 1 - Head 2)
    scale = epsilon
    scores = torch.bmm(Q, K.transpose(1, 2)) * scale
    attn_probs = torch.softmax(scores, dim=-1)
    head1 = torch.bmm(attn_probs, V)
    
    mean_v = V.mean(dim=1, keepdim=True)
    head2 = mean_v.expand_as(head1)
    
    # Full correction: (H1 - H2) * (N/eps)
    approx_full = (head1 - head2) * (n / epsilon)
    
    # Ablated Model (Head 1 only)
    # Dropping the aggregation head
    approx_ablated = head1 * (n / epsilon)
    
    # Exact Target K V
    # Note: K is unnormalized dot product here
    exact = torch.bmm(torch.bmm(Q, K.transpose(1, 2)), V)
    
    # Errors
    def rel_err(pred, true):
        return float(torch.norm(pred - true) / torch.norm(true))
    
    err_full = rel_err(approx_full, exact)
    err_ablated = rel_err(approx_ablated, exact)
    
    return {
        "err_full": err_full,
        "err_ablated": err_ablated,
        "improvement_factor": err_ablated / (err_full + 1e-9)
    }

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", type=str, default="docs/figures/ablations/route_b_heads.png")
    args, _ = parser.parse_known_args()
    
    plot = args.plot or cfg.get("plot", False)
    out_path = args.out or cfg.get("out", "docs/figures/ablations/route_b_heads.png")

    res = run_head_drop_ablation(
        seed=int(cfg.get("seed", 123)),
        n=int(cfg.get("n_support", 64)),
        p=int(cfg.get("p", 16)),
        epsilon=float(cfg.get("epsilon", 1e-4))
    )
    print(json.dumps(res))
    
    if plot:
        try:
            plt.figure(figsize=(6, 5))
            labels = ['Full Model (2 Heads)', 'Ablated (Head 1 Only)']
            vals = [res['err_full'], res['err_ablated']]
            colors = ['green', 'red']
            
            plt.bar(labels, vals, color=colors, alpha=0.7)
            plt.ylabel('Relative Approximation Error')
            plt.title('Route B Construction: Impact of Aggregation Head')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot to {out_path}")
        except Exception as e:
            print(f"Error plotting: {e}")

if __name__ == "__main__":
    main()
