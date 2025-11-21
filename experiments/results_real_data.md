# Real-World ICL Evaluation Results

We evaluated the In-Context Learning (ICL) capabilities of our Ridge Regression baseline (using Bag-of-Words features) on three real-world text datasets: **20 Newsgroups** (binary subset), **STS-B** (small sample), and **TREC-6** (question classification).

## Methodology
- **Model**: Ridge Regression ($\alpha=1.0$) on unigram Bag-of-Words features.
- **Tokenizer**: Simple numeric tokenizer with greedy vocabulary building.
- **Metric**: $R^2$ score and Mean Squared Error (MSE) on held-out query sets.
- **Protocol**: For each support size $k$, we sample $k$ support examples and $n_{query}$ query examples, repeat 3 times with different seeds, and report mean $\pm$ std.

## Results Summary

### 1. 20 Newsgroups (Binary Classification)
*Task: Distinguish between 'sci.space' and 'talk.politics.misc'.*

| Support ($k$) | $R^2$ (Mean) | MSE (Mean) |
| :--- | :--- | :--- |
| 4 | -0.04 | 0.25 |
| 8 | -0.22 | 0.29 |
| 16 | -0.23 | 0.30 |
| 32 | -1.21 | 0.51 |
| 64 | -3.94 | 1.21 |
| 128 | -1.94 | 0.72 |

**Observation**: The model fails to learn effectively with this simple BoW representation on this task, actually degrading as support increases (likely due to overfitting the support set with a growing vocabulary that doesn't generalize well to queries in this high-dimensional sparse space).

### 2. STS-B (Semantic Similarity)
*Task: Predict similarity score (0-5) between sentence pairs.*

| Support ($k$) | $R^2$ (Mean) | MSE (Mean) |
| :--- | :--- | :--- |
| 4 | -0.01 | 2.75 |
| 8 | -0.41 | 4.22 |
| 16 | -0.72 | 3.65 |
| 20 | **0.00** | 2.57 |

**Observation**: Performance is near zero or negative $R^2$. The manual sample size is too small, and BoW is a poor feature set for semantic similarity (which relies on word order and synonyms).

### 3. TREC-6 (Question Classification)
*Task: Classify questions into 6 categories (DESC, ENTY, ABBR, HUM, NUM, LOC).*

| Support ($k$) | $R^2$ (Mean) | MSE (Mean) |
| :--- | :--- | :--- |
| 4 | -0.16 | 124.36 |
| 32 | -0.21 | 115.78 |
| 256 | **0.06** | 94.16 |

**Observation**: We see a slight improvement at $k=256$, achieving a positive $R^2$ (0.06).

### 4. AG News (News Topic Classification)
*Task: Classify news articles into 4 topics (World, Sports, Business, Sci/Tech).*

| Support ($k$) | $R^2$ (Mean) | MSE (Mean) |
| :--- | :--- | :--- |
| 4 | -0.34 | 1.74 |
| 16 | -0.05 | 1.39 |
| 64 | **0.13** | 1.02 |
| 128 | **0.21** | 0.94 |
| 256 | **0.13** | 1.02 |

**Observation**: AG News shows the most promise with BoW features. We see a clear trend where performance improves as support size increases, reaching a peak $R^2 \approx 0.21$ at $k=128$. This indicates that the "Topic" signal in AG News is robust enough to be picked up by a simple Ridge ICL baseline even without pre-trained embeddings, provided sufficient support examples.

## Conclusion
The current "ICL" baseline using Ridge Regression on raw Bag-of-Words features struggles with these few-shot tasks. This highlights the need for:
1. **Better Representations**: Using pre-trained embeddings (e.g., BERT, GPT) instead of raw BoW would likely yield much better few-shot performance.
2. **Meta-Learning**: The current setup is "training from scratch" on the context. True Meta-ICL involves a model *pretrained* to do this learning. Our project investigates the mechanism of this (Linear Attention $\approx$ GD/Ridge), but our baseline here uses raw features which are hard to learn quickly.

The code infrastructure `experiments/real_data_eval.py` is robust and can easily be swapped to use embeddings if we add a dependency on `transformers`.

