# LAT→CG Constructive Proof (Sketch)

Goal: Show a decoder-style transformer with linear attention (unnormalized dot-products) over k support tokens + 1 query + 1 aggregator simulates t steps of Conjugate Gradient (CG) for (K+λI)α=y with K=ΦΦ^T, outputting f_t(x)=k(x)^T α_t.

Construction:

- Tokens: k supports carry per-token scalars/channels for α_i, r_i, p_i; one aggregator for global reductions; one query for final readout.
- Linear attention head (Lemma A): with q_j=φ(x_j), k_i=φ(x_i), v_i=p_i, the attention value at token j computes (Kp)_j = ∑_i ⟨φ_j, φ_i⟩ p_i.
- Aggregator (Lemma B): an aggregator token, attending to supports, computes reductions ∑_i r_i^2 and ∑_i p_i (Ap)_i and broadcasts the resulting scalars back to supports.
- Token-wise affine updates (Lemma C): a per-token MLP with residual updates performs α←α+γ p, r←r−γ Ap, p←r+β p using the broadcast γ, β.

Causal mask (S5):

- Compute phase: supports↔supports and aggregator→support enabled; query isolated.
- Readout phase: query→supports (and aggregator) enabled; supports do not read query.

Correctness:

- Each CG step requires: two reductions (rr, pAp), one mat-vec (Kp), and three affine updates. The heads/MLP implement these operations exactly at finite precision.
- Adding λ is a per-token residual add (λ p) folded into Ap.

Rate bound:

- Standard CG rate applies: \( ||α_t−α_*||_{K+λI} ≤ 2((√κ−1)/(√κ+1))^t ||α_*||_{K+λI} \) with κ=cond(K+λI), hence prediction error \(|f_t(x)−f_*(x)| ≤ √{k(x)^T (K+λI)^{-1} k(x)}×2((√κ−1)/(√κ+1))^t ||α_*||_{K+λI}.\)

Softmax bridge:

- Replace linear attention logits with softmax logits S/τ; exp(S) recovers an exponential kernel; normalization handled via aggregator or auxiliary channels.

Notes:

- Width m≥p+c suffices to carry φ and CG channels; m<p induces a rank-m sketch K̂ with spectral-tail bias (see width–rank note).
- Numerical stability: float32 runtime with double-precision tests; ε-machine noise leads to negligible drift under bounded t.
