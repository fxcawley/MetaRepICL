import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path

try:
    from experiments.real_data_transformer import run_transformer_icl
    from experiments.real_data_eval import run_real_data_eval
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.real_data_transformer import run_transformer_icl
    from experiments.real_data_eval import run_real_data_eval

def run_comparison(
    csv_path: str,
    n_support_list=[8, 16, 32, 64, 128],
    seeds=[1, 2, 3]
):
    results = []
    
    for k in n_support_list:
        ridge_mses = []
        trans_mses = []
        
        for s in seeds:
            # Ridge (BoW + Linear Regression)
            r_ridge = run_real_data_eval(
                csv_path, n_support=k, n_query=32, seed=s
            )
            if "mse" in r_ridge:
                ridge_mses.append(r_ridge["mse"])
                
            # Transformer (BoW + Softmax Attention)
            # Note: Softmax attention is effectively a kernel smoother.
            # With high tau, it averages. With low tau, it matches nearest neighbors.
            # We assume a tuned tau (e.g. 5.0 for normalized vectors) or just use default.
            r_trans = run_transformer_icl(
                csv_path, n_support=k, n_query=32, vocab_size=2000, tau=5.0, seed=s, plot_attention=False
            )
            if "mse" in r_trans:
                trans_mses.append(r_trans["mse"])
                
        results.append({
            "n_support": k,
            "ridge_mse": np.mean(ridge_mses),
            "ridge_std": np.std(ridge_mses),
            "trans_mse": np.mean(trans_mses),
            "trans_std": np.std(trans_mses)
        })
        
    return results

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cwd = Path(__file__).resolve().parent
    repo_root = cwd.parent
    
    data_path = repo_root / "data" / "ag_news_train.csv"
    if not data_path.exists():
        print("AG News data not found.")
        return
        
    print("Running comparison on AG News...")
    res = run_comparison(str(data_path))
    
    # Plot
    os.makedirs("figures/comparison", exist_ok=True)
    ks = [r["n_support"] for r in res]
    mse_ridge = [r["ridge_mse"] for r in res]
    mse_trans = [r["trans_mse"] for r in res]
    
    plt.figure()
    plt.plot(ks, mse_ridge, marker='o', label='Ridge (Linear)')
    plt.plot(ks, mse_trans, marker='s', label='Transformer (Softmax)')
    plt.xlabel("Number of Support Examples")
    plt.ylabel("MSE (Lower is better)")
    plt.title("ICL Performance: Ridge vs Softmax Attention (AG News)")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/comparison/ag_news_comparison.png")
    
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

