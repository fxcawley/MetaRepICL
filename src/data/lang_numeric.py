import numpy as np
import torch
from typing import List, Tuple, Dict

class NumericTokenizer:
    """
    A simple tokenizer that treats numeric labels as tokens.
    Maps input words to IDs and numeric labels to specific numeric token IDs.
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word2id = {}
        self.id2word = {}
        self.next_id = 0
        
        # Reserve special tokens?
        self.pad_token = "[PAD]"
        self.add_token(self.pad_token)
        
    def add_token(self, word: str) -> int:
        if word not in self.word2id:
            self.word2id[word] = self.next_id
            self.id2word[self.next_id] = word
            self.next_id += 1
        return self.word2id[word]
    
    def encode(self, text: str) -> List[int]:
        ids = []
        for word in text.split():
            if word not in self.word2id:
                if self.next_id < self.vocab_size:
                    self.add_token(word)
                else:
                    # Hash collision / OOV strategy for toy tokenizer
                    # Just map to hash bucket
                    ids.append(hash(word) % self.vocab_size)
                    continue
            ids.append(self.word2id[word])
        return ids

def make_lang_numeric_dataset(
    n_support: int,
    n_query: int,
    n_tokens_per_doc: int = 10,
    seed: int = 123
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """
    Generate synthetic 'language' tasks where the task is to predict a numeric label
    associated with a document.
    
    Here we simulate a scenario where the label is a function of specific keywords in the document.
    E.g. "good", "excellent" -> positive score.
    
    Returns:
        (docs_support, labels_support, docs_query, labels_query)
    """
    rng = np.random.default_rng(seed)
    
    vocab = ["the", "a", "an", "movie", "is", "was", "this", "that", "it", 
             "good", "great", "excellent", "amazing", # Positive words
             "bad", "terrible", "awful", "horrible", # Negative words
             "okay", "average", "so-so", "fine"]     # Neutral
             
    positive_weights = {
        "good": 1.0, "great": 2.0, "excellent": 3.0, "amazing": 4.0
    }
    negative_weights = {
        "bad": -1.0, "terrible": -2.0, "awful": -3.0, "horrible": -4.0
    }
    
    def generate_doc_and_label() -> Tuple[str, float]:
        # Sample random words
        doc_words = rng.choice(vocab, size=n_tokens_per_doc, replace=True)
        
        # Calculate score
        score = 0.0
        for w in doc_words:
            score += positive_weights.get(w, 0.0)
            score += negative_weights.get(w, 0.0)
            
        return " ".join(doc_words), score

    docs_s, ys = [], []
    for _ in range(n_support):
        d, y = generate_doc_and_label()
        docs_s.append(d)
        ys.append(y)
        
    docs_q, yq = [], []
    for _ in range(n_query):
        d, y = generate_doc_and_label()
        docs_q.append(d)
        yq.append(y)
        
    return docs_s, np.array(ys), docs_q, np.array(yq)

