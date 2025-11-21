import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from sklearn.preprocessing import normalize

try:
    from src.data.lang_numeric import load_dataset_from_csv, NumericTokenizer
except ImportError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.data.lang_numeric import load_dataset_from_csv, NumericTokenizer

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

def run_transformer_icl(
    csv_path: str,
    n_support: int = 32,
    n_query: int = 1,
    vocab_size: int = 1000,
    tau: float = 10.0, # Temperature for softmax
    seed: int = 123,
    plot_attention: bool = False,
    out_dir: str = "figures/attention"
):
    """
    Runs a single-layer Softmax Attention block on real text data (BoW features).
    Visualizes the attention weights.
    """
    # Load Data
    docs_s, ys, docs_q, yq = load_dataset_from_csv(
        csv_path, n_support, n_query, seed=seed
    )
    
    if len(docs_s) == 0:
        return {"error": "No data"}

    # Featurize (BoW)
    tokenizer = NumericTokenizer(vocab_size=vocab_size)
    
    def vectorize(docs):
        X = np.zeros((len(docs), tokenizer.vocab_size))
        for i, d in enumerate(docs):
            ids = tokenizer.encode(d)
            for idx in ids:
                X[i, idx] += 1
        return X

    Xs_np = vectorize(docs_s)
    Xq_np = vectorize(docs_q)
    
    # Normalize features? Transformers usually work on normalized embeddings (LayerNorm)
    # For BoW, L2 norm makes them cosine-similarity ready
    Xs_np = normalize(Xs_np, axis=1)
    Xq_np = normalize(Xq_np, axis=1)
    
    # Convert to Torch
    # Dimensions: (Batch=1, Seq, Dim)
    # We treat the context as a sequence of [Support_1, ..., Support_k, Query]
    # But standard ICL is: predict for Query attending to Supports.
    # Q = Xq, K = Xs, V = ys
    
    Q = torch.tensor(Xq_np, dtype=torch.float32).unsqueeze(0) # (1, n_query, d)
    K = torch.tensor(Xs_np, dtype=torch.float32).unsqueeze(0) # (1, n_support, d)
    V = torch.tensor(ys, dtype=torch.float32).unsqueeze(0).unsqueeze(2) # (1, n_support, 1)
    
    # Attention
    # Score = Q @ K^T / tau
    scores = torch.bmm(Q, K.transpose(1, 2)) / tau # (1, n_query, n_support)
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Output = Weights @ V
    y_pred = torch.bmm(attn_weights, V).squeeze() # (n_query,) or scalar if n_query=1
    
    # Eval
    if n_query == 1:
        y_pred = y_pred.unsqueeze(0)
        
    mse = np.mean((y_pred.detach().numpy() - yq)**2)
    
    # Visualization
    if plot_attention:
        os.makedirs(out_dir, exist_ok=True)
        
        # Plot attention weights for the first query
        w = attn_weights[0, 0, :].detach().numpy() # (n_support,)
        
        plt.figure(figsize=(10, 4))
        plt.bar(range(n_support), w)
        plt.xlabel('Support Example Index')
        plt.ylabel('Attention Weight')
        plt.title(f'Attention Weights (MSE={mse:.4f})')
        
        # Annotate with labels
        for i, val in enumerate(w):
            label = ys[i]
            plt.text(i, val, f"y={label:.0f}", ha='center', va='bottom', fontsize=8)
            
        plt.tight_layout()
        plt.savefig(f"{out_dir}/attn_seed{seed}.png")
        plt.close()
        
    return {
        "mse": float(mse),
        "n_support": n_support,
        "tau": tau
    }

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cwd = Path(__file__).resolve().parent
    repo_root = cwd.parent
    
    # Default to AG News as it worked best
    path_str = cfg.get("data_path", "data/ag_news_train.csv")
    if Path(path_str).is_absolute():
        data_path = Path(path_str)
    else:
        data_path = repo_root / path_str
        
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return

    res = run_transformer_icl(
        str(data_path),
        n_support=int(cfg.get("n_support", 32)),
        n_query=int(cfg.get("n_query", 1)), # Visualization assumes 1 query usually
        vocab_size=int(cfg.get("vocab_size", 2000)),
        tau=float(cfg.get("tau", 5.0)), # Tunable temp
        seed=int(cfg.get("seed", 123)),
        plot_attention=True
    )
    print(json.dumps(res))

if __name__ == "__main__":
    main()

