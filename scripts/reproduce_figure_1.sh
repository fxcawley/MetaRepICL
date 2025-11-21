#!/bin/bash
# Reproduces Key Figure 1: Route A Softmax KRR alignment (MVP)
# and generates the Failure Mode plot (Figure 2 in paper).

set -e

echo "=================================================="
echo "MetaRep Reproducibility Script"
echo "Generating Figure 1: Route A MVP (Softmax vs Oracle)"
echo "=================================================="

# Run Route A Minimal
# Use Hydra overrides for plot and out
python experiments/route_a_minimal.py ++plot=true ++out=figures/route_a_mvp.png seed=123

echo "Generated figures/route_a_mvp.png"

echo "=================================================="
echo "Generating Figure 2: Failure Modes (CG Stall)"
echo "=================================================="

# Run Failure Mode Demo
# This script is pure python/hydra independent in main() currently, assuming fixed path
python experiments/failure_modes/ill_conditioned.py

echo "Generated figures/failure_modes/ill_conditioned_cg.png"

echo "=================================================="
echo "Generating Figure 3: Ablations (Head Sharing)"
echo "=================================================="

# Run Ablations
python experiments/ablations/heads_temp.py ++plot=true

echo "Generated figures/ablations/route_b_heads.png"
echo "Generated figures/ablations/route_a_temp.png"

echo "=================================================="
echo "All key figures reproduced successfully."
echo "=================================================="
