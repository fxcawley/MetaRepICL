import json
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from sklearn.linear_model import Ridge

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

def run_real_data_eval(
    csv_path: str,
    n_support: int = 8,
    n_query: int = 4,
    seed: int = 123
):
    docs_s, ys, docs_q, yq = load_dataset_from_csv(
        csv_path, n_support, n_query, seed=seed
    )
    
    # Check if we loaded data
    if len(docs_s) == 0:
        return {"error": "No data loaded"}
        
    tokenizer = NumericTokenizer(vocab_size=2000)
    # Pre-populate tokenizer with some vocab from support to ensure overlap?
    # Tokenizer builds vocab on the fly if we call it.
    # But 'encode' adds to vocab if we allow it.
    # We should fit vocab on support set?
    # The current toy tokenizer adds to vocab greedily.
    # This is effectively "training" the tokenizer on the support set if we run it first.
    
    def vectorize(docs, fit=False):
        # First pass to build vocab if fitting? 
        # Actually, the tokenizer updates state.
        # We should update on support docs.
        # But for query docs, we should not update vocab?
        # The simple tokenizer just hashes OOV or adds.
        # Let's just run it.
        X = np.zeros((len(docs), tokenizer.vocab_size))
        for i, d in enumerate(docs):
            ids = tokenizer.encode(d)
            for idx in ids:
                X[i, idx] += 1
        return X
        
    # "Train" tokenizer on support
    Xs = vectorize(docs_s)
    # "Eval" on query (vocab is fixed... wait, encode() adds new words if not found?)
    # We need to freeze tokenizer or handle OOV.
    # The toy tokenizer `encode` adds tokens if `next_id < vocab_size`.
    # So query words will be added to vocab. This is leakage-ish if we consider vocab structure,
    # but for bag-of-words Ridge, it just adds features.
    # If a word appears in query but not support, its column in Xs will be all 0.
    # The weight for that column will be 0 (Ridge penalty).
    # So it has no effect. This is fine.
    
    Xq = vectorize(docs_q)
    
    # Run Ridge
    model = Ridge(alpha=1.0)
    model.fit(Xs, ys)
    r2 = model.score(Xq, yq)
    preds = model.predict(Xq)
    mse = np.mean((preds - yq)**2)
    
    return {
        "r2": r2,
        "mse": mse,
        "n_support": n_support,
        "n_query": n_query,
        "vocab_size": tokenizer.next_id
    }

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cwd = Path(__file__).resolve().parent
    repo_root = cwd.parent
    
    # Allow overriding data path
    path_str = cfg.get("data_path", "data/sentiment.csv")
    if Path(path_str).is_absolute():
        data_path = Path(path_str)
    else:
        data_path = repo_root / path_str
    
    if not data_path.exists():
        # Fallback for legacy default location if not found
        fallback = repo_root / "data" / "sentiment.csv"
        if fallback.exists() and "data_path" not in cfg:
            data_path = fallback
        else:
            print(f"Data file not found at {data_path}")
            return

    print(f"Evaluating on {data_path}...")
    res = run_real_data_eval(
        str(data_path),
        n_support=int(cfg.get("n_support", 8)),
        n_query=int(cfg.get("n_query", 4)),
        seed=int(cfg.get("seed", 123))
    )
    print(json.dumps(res))

if __name__ == "__main__":
    main()

