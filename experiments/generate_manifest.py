import json
import glob
import os
from datetime import datetime

def generate_manifest():
    """
    Aggregates all experiment results and dataset metadata into a single manifest file.
    """
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "experiments": [],
        "datasets": []
    }
    
    # 1. Datasets
    data_files = glob.glob("data/*.csv")
    for f in data_files:
        try:
            # Count lines roughly
            with open(f, 'r', encoding='utf-8') as fp:
                lines = sum(1 for _ in fp) - 1 # Header
            manifest["datasets"].append({
                "name": os.path.basename(f),
                "path": f,
                "size": lines,
                "type": "csv"
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 2. Experiments
    # We don't have saved JSON logs for every run unless we parse stdout or rely on Hydra logs.
    # But we can structure what we have: the 'real_data_sweep' output is usually printed.
    # Let's assume we want to save the latest sweep result if available, or just define what *can* be run.
    
    # For this MVP, let's create a manifest of the *capabilities* and where to find results.
    manifest["experiments"] = [
        {
            "name": "baselines",
            "command": "python experiments/run_eval.py target=baselines",
            "description": "Ridge Oracle and GD-ICL baselines on synthetic linear data."
        },
        {
            "name": "width_rank",
            "command": "python experiments/run_eval.py target=width_rank",
            "description": "Effect of width m on prediction error (Spectral Tail)."
        },
        {
            "name": "route_a",
            "command": "python experiments/run_eval.py target=route_a",
            "description": "Softmax Route A vs Exponential Kernel KRR."
        },
        {
            "name": "route_b",
            "command": "python experiments/run_eval.py target=route_b",
            "description": "Route B approximation error (Two-head rescaling)."
        },
        {
            "name": "precond",
            "command": "python experiments/run_eval.py target=precond",
            "description": "Diagonal preconditioning on ill-conditioned data."
        },
        {
            "name": "probes",
            "command": "python experiments/run_eval.py target=probes",
            "description": "Linear probes for recovering CG state."
        },
        {
            "name": "real_data_sweep",
            "command": "python experiments/real_data_sweep.py",
            "description": "Support size sweeps on real-world text datasets (20News, TREC, AG News, STS-B)."
        }
    ]
    
    with open("output_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Manifest generated at output_manifest.json")

if __name__ == "__main__":
    generate_manifest()

