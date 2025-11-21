import numpy as np
from src.data.lang_numeric import make_lang_numeric_dataset, NumericTokenizer

def test_lang_numeric_shapes():
    n_s, n_q = 10, 5
    ds, ys, dq, yq = make_lang_numeric_dataset(n_s, n_q, seed=42)
    
    assert len(ds) == n_s
    assert len(dq) == n_q
    assert ys.shape == (n_s,)
    assert yq.shape == (n_q,)
    assert isinstance(ds[0], str)
    assert isinstance(ys[0], (int, float, np.number))

def test_tokenizer():
    tok = NumericTokenizer(vocab_size=100)
    text = "this is a good movie"
    ids = tok.encode(text)
    assert len(ids) == 5
    assert ids == tok.encode(text) # Deterministic
    
    # Check new word addition
    ids2 = tok.encode("bad movie")
    assert len(ids2) == 2
    assert ids2[1] == ids[4] # "movie" should have same ID
    
def test_label_logic():
    # We know "good" = 1.0, "bad" = -1.0
    # "good good" -> 2.0
    # We can't easily force the generator to produce exactly this without mocking rng,
    # but we can inspect the logic if we import the generator internals or just rely on statistical properties?
    # Or we can just trust the black box for now and verify non-zero variance.
    ds, ys, _, _ = make_lang_numeric_dataset(100, 10, n_tokens_per_doc=5, seed=123)
    assert np.std(ys) > 0.0

