# Project Status and Interpretations

## Current Status
As of late 2025, the **MetaRep** project has successfully:
1.  **Formalized the Theory**: We have constructive proofs for mapping Softmax Attention to Exponential KRR and Linear Attention to PCG.
2.  **Validated on Synthetic Data**: Our "minimal" experiments (`experiments/route_a_minimal.py`, `experiments/width_rank.py`) confirm that small Transformers *can* be trained to implement these algorithms with high fidelity.
3.  **Identified Mechanistic Signatures**: We found that linear probes can recover the specific intermediate variables of the CG algorithm ($\alpha, r, p$) from the residual stream.

## Honest Interpretations

### What works well
-   **The KRR Connection**: The link between Softmax attention and the Exponential Kernel is robust and appears to be the dominant mode of operation for small-context dense attention models.
-   **Preconditioning**: The theory that normalization layers (LayerNorm) act as diagonal preconditioners is supported by our failure-mode experiments.

### Limitations and Risks
-   **Synthetic vs. Real**: All mechanistic evidence is currently derived from synthetic linear regression tasks. While we have `real_data` loaders, we have not yet confirmed if Large Language Models (LLMs) trained on text use *this specific* algorithm or a more heuristic variant.
-   **Route B Complexity**: The Route B (PCG) construction is complex and sensitive to head configurations (as shown in our ablations). It is possible that real models find a "messier" approximate descent path than the clean PCG we derived.

## Future Work
1.  **Scaling to LLMs**: Apply our `state_probes` to LLaMA-2-7B on in-context learning benchmarks (MMLU).
2.  **Non-Linear Tasks**: Extend the theory to Generalized Linear Models (GLMs) where the loss surface is convex but not quadratic (Logistic Regression).
3.  **Causal Interventions**: Move beyond probing to *intervening*—can we inject a "better" search direction $p_t$ into the residual stream and speed up ICL?

---
*This project is open-source under the MIT License. We welcome contributions.*


# Project Status and Interpretations

## Current Status
As of late 2025, the **MetaRep** project has successfully:
1.  **Formalized the Theory**: We have constructive proofs for mapping Softmax Attention to Exponential KRR and Linear Attention to PCG.
2.  **Validated on Synthetic Data**: Our "minimal" experiments (`experiments/route_a_minimal.py`, `experiments/width_rank.py`) confirm that small Transformers *can* be trained to implement these algorithms with high fidelity.
3.  **Identified Mechanistic Signatures**: We found that linear probes can recover the specific intermediate variables of the CG algorithm ($\alpha, r, p$) from the residual stream.

## Honest Interpretations

### What works well
-   **The KRR Connection**: The link between Softmax attention and the Exponential Kernel is robust and appears to be the dominant mode of operation for small-context dense attention models.
-   **Preconditioning**: The theory that normalization layers (LayerNorm) act as diagonal preconditioners is supported by our failure-mode experiments.

### Limitations and Risks
-   **Synthetic vs. Real**: All mechanistic evidence is currently derived from synthetic linear regression tasks. While we have `real_data` loaders, we have not yet confirmed if Large Language Models (LLMs) trained on text use *this specific* algorithm or a more heuristic variant.
-   **Route B Complexity**: The Route B (PCG) construction is complex and sensitive to head configurations (as shown in our ablations). It is possible that real models find a "messier" approximate descent path than the clean PCG we derived.

## Future Work
1.  **Scaling to LLMs**: Apply our `state_probes` to LLaMA-2-7B on in-context learning benchmarks (MMLU).
2.  **Non-Linear Tasks**: Extend the theory to Generalized Linear Models (GLMs) where the loss surface is convex but not quadratic (Logistic Regression).
3.  **Causal Interventions**: Move beyond probing to *intervening*—can we inject a "better" search direction $p_t$ into the residual stream and speed up ICL?

---
*This project is open-source under the MIT License. We welcome contributions.*

