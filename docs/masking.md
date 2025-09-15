# Causal Masking Proof Obligations (S5)

We use a fixed token layout: k support tokens, 1 query token, 1 aggregator token. Two phases:

- compute: construct CG reductions/state using only support↔support and aggregator↔support communication; query is isolated.
- readout: query reads from supports (and optionally aggregator) to produce prediction; supports do not read from query.

Mask semantics (mask[i,j] = -∞ blocks attention, 0 allows):

- compute phase:
  - support→{support, aggregator} allowed
  - aggregator→support allowed
  - query→{support, aggregator} blocked
  - aggregator→query blocked (no backchannel)
- readout phase:
  - query→{support, aggregator} allowed
  - support→query blocked

Obligations:

1. No query→support (or support→query) paths exist in compute phase (no leakage of y).
2. Aggregator does not route query information back to supports before readout.
3. Readout occurs in a final phase after CG state is computed.

See `src/lat/masking.py` for mask construction and tests for invariants.
