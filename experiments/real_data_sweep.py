import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path

try:
    from experiments.real_data_eval import run_real_data_eval
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from experiments.real_data_eval import run_real_data_eval

def run_support_sweep(data_path, supports=[4, 8, 16, 32, 64], n_query=16, seeds=[1, 2, 3]):
    results = []
    
    df = pd.read_csv(data_path)
    total_len = len(df)
    
    for k in supports:
        if k + n_query > total_len:
            continue
            
        seed_r2 = []
        seed_mse = []
        
        for s in seeds:
            res = run_real_data_eval(data_path, n_support=k, n_query=n_query, seed=s)
            if "r2" in res:
                seed_r2.append(res["r2"])
                seed_mse.append(res["mse"])
        
        if seed_r2:
            results.append({
                "n_support": k,
                "r2_mean": np.mean(seed_r2),
                "r2_std": np.std(seed_r2),
                "mse_mean": np.mean(seed_mse),
                "mse_std": np.std(seed_mse)
            })
            
    return results

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cwd = Path(__file__).resolve().parent
    repo_root = cwd.parent
    
    datasets = {
        "20news": repo_root / "data" / "20news_binary.csv",
        "stsb": repo_root / "data" / "stsb_sample.csv",
        "trec": repo_root / "data" / "trec_train.csv"
    }
    
    os.makedirs("figures/real_data", exist_ok=True)
    
    all_results = {}
    
    for name, path in datasets.items():
        if not path.exists():
            print(f"Skipping {name}, not found at {path}")
            continue
            
        print(f"Sweeping {name}...")
        # Adjust sweep range based on dataset size
        if name == "stsb":
            supports = [4, 8, 16, 20] # Small manual dataset
            n_query = 8
        elif name == "trec":
            supports = [4, 8, 16, 32, 64, 128, 256]
            n_query = 64
        else:
            supports = [4, 8, 16, 32, 64, 128]
            n_query = 32
            
        res = run_support_sweep(str(path), supports=supports, n_query=n_query)
        all_results[name] = res
        
        # Plot
        ks = [r["n_support"] for r in res]
        r2s = [r["r2_mean"] for r in res]
        errs = [r["r2_std"] for r in res]
        
        plt.figure()
        plt.errorbar(ks, r2s, yerr=errs, marker='o', capsize=5)
        plt.xlabel("Number of Support Examples")
        plt.ylabel("R^2 Score")
        plt.title(f"ICL Performance on {name}")
        plt.grid(True)
        plt.savefig(f"figures/real_data/{name}_sweep.png")
        plt.close()
        
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()

