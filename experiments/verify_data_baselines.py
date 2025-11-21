import json
import numpy as np
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.glm import make_glm_dataset
from src.data.lang_numeric import make_lang_numeric_dataset, NumericTokenizer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge

def run_glm_baselines(
    n_support: int = 64,
    n_query: int = 32,
    p: int = 16,
    seed: int = 123
):
    # Logistic
    Xs, ys, Xq, yq = make_glm_dataset(n_support, n_query, p, 'logistic', seed=seed)
    # Sklearn > 1.2 deprecated 'none', use None
    model = LogisticRegression(penalty=None, fit_intercept=False) # Assume zero-centered data
    # Sklearn needs 2 classes for binary
    if len(np.unique(ys)) < 2:
        # Fallback if random generation produced only 1 class
        score_log = 0.0
    else:
        model.fit(Xs, ys)
        score_log = model.score(Xq, yq) # Accuracy for classification
        
    # Poisson
    Xs, ys, Xq, yq = make_glm_dataset(n_support, n_query, p, 'poisson', seed=seed)
    model = PoissonRegressor(alpha=0.0, fit_intercept=False)
    model.fit(Xs, ys)
    score_poi = model.score(Xq, yq) # D^2 score (explained deviance)
    
    return {"logistic_acc": score_log, "poisson_d2": score_poi}

def run_lang_baselines(
    n_support: int = 64,
    n_query: int = 32,
    seed: int = 123
):
    docs_s, ys, docs_q, yq = make_lang_numeric_dataset(n_support, n_query, seed=seed)
    
    # Simple Bag-of-Words features
    tokenizer = NumericTokenizer(vocab_size=1000)
    
    def vectorize(docs):
        X = np.zeros((len(docs), tokenizer.vocab_size))
        for i, d in enumerate(docs):
            ids = tokenizer.encode(d)
            for idx in ids:
                X[i, idx] += 1
        return X
        
    Xs = vectorize(docs_s)
    Xq = vectorize(docs_q)
    
    # Ridge Regression Baseline
    model = Ridge(alpha=1.0)
    model.fit(Xs, ys)
    r2 = model.score(Xq, yq)
    
    return {"lang_ridge_r2": r2}

def main():
    res_glm = run_glm_baselines()
    res_lang = run_lang_baselines()
    print(json.dumps({**res_glm, **res_lang}))

if __name__ == "__main__":
    main()

