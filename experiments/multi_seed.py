"""Multi-Seed Robustness Check for Algorithm Identification.

Runs the algorithm identification pipeline across multiple seeds to
demonstrate that findings are robust, not a single-seed artifact.

For each seed: train model (or load checkpoint), train per-layer readouts,
run full algorithm_id analysis, store per-seed weighted R² and gaps.

Outputs:
  - multi_seed_summary.json   -- per-seed + aggregate results
  - multi_seed_ranking.png    -- mean R² per algo with std error bars
  - multi_seed_gap.png        -- GD gap and top-3 gap per seed

Usage:
    python experiments/multi_seed.py --seeds 42 123 456
    python experiments/multi_seed.py --load-dir docs/figures/trained_mixed
    python experiments/multi_seed.py --steps 5000 --seeds 42 123
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.icl_transformer import ICLTransformer
from algorithm_id import (
    ALGORITHMS, ALGO_COLORS, KAPPAS,
    generate_batch_with_cov, generate_mixed_kappa_batch,
    train_model, train_layer_readouts, run_algorithm_identification,
)

# ---------------------------------------------------------------------------
# Per-seed analysis helpers
# ---------------------------------------------------------------------------

def compute_weighted_r2(results):
    """Kappa-weighted mean R² per algorithm (same logic as algorithm_id.py)."""
    weighted_r2 = {}
    norm = sum(math.log10(max(k, 1.1)) for k in results)
    for name in ALGORITHMS:
        total = sum(results[k]['mean_r2'][name] * math.log10(max(k, 1.1))
                    for k in results)
        weighted_r2[name] = total / norm
    return weighted_r2


def compute_gaps(weighted_r2):
    """Return (gd_gap, top3_gap, ranking) from weighted R² dict."""
    sorted_algos = sorted(weighted_r2, key=lambda n: -weighted_r2[n])
    gd_gap = weighted_r2[sorted_algos[0]] - weighted_r2['GD']
    top3_gap = (weighted_r2[sorted_algos[0]] - weighted_r2[sorted_algos[2]]
                if len(sorted_algos) >= 3 else 0.0)
    return gd_gap, top3_gap, sorted_algos

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ranking_bars(algo_names, mean_r2, std_r2, out_dir):
    """Bar chart of mean weighted R² per algorithm with std error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(algo_names))
    colors = [ALGO_COLORS[n] for n in algo_names]
    ax.bar(x, [mean_r2[n] for n in algo_names], yerr=[std_r2[n] for n in algo_names],
           color=colors, alpha=0.85, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, fontsize=9)
    ax.set_ylabel('Kappa-weighted mean R²')
    ax.set_title('Multi-Seed Algorithm Identification:\n'
                 'Mean ± Std of Weighted R² Across Seeds',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for i, n in enumerate(algo_names):
        ax.text(i, mean_r2[n] + std_r2[n] + 0.02,
                f'{mean_r2[n]:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/multi_seed_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_gap_per_seed(seeds, gd_gaps, top3_gaps, out_dir):
    """Per-seed plot of the GD gap and top-3 gap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(seeds))
    labels = [str(s) for s in seeds]

    for ax, gaps, color, ylabel, title in [
        (axes[0], gd_gaps, 'steelblue', 'Best R² − GD R²', 'Model >> GD Gap Per Seed'),
        (axes[1], top3_gaps, 'darkorange', '1st R² − 3rd R²', 'Top-3 Gap Per Seed'),
    ]:
        ax.bar(x, gaps, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axhline(np.mean(gaps), color='red', linestyle='--', linewidth=1.5,
                   label=f'mean={np.mean(gaps):.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel('Seed')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Multi-Seed Robustness: Gap Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/multi_seed_gap.png', dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Seed Robustness Check for Algorithm Identification')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2024],
                        help='Seeds to evaluate (default: 42 123 456 789 2024)')
    parser.add_argument('--steps', type=int, default=50000, help='Training steps per seed')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-support', type=int, default=20)
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--n-per-kappa', type=int, default=200)
    parser.add_argument('--out-dir', type=str, default='docs/figures/multi_seed')
    parser.add_argument('--load-dir', type=str, default=None,
                        help='Directory with model_seed_{s}.pt checkpoint files')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.out_dir}/")

    cfg = {
        'steps': args.steps, 'batch_size': args.batch_size,
        'n_support': args.n_support, 'p': args.p,
        'noise': args.noise, 'lr': args.lr, 'wd': 0.01,
        'warmup': min(1000, args.steps // 10),
        'log_every': max(1, args.steps // 25),
        'probe_lam': 0.1,
    }
    input_dim = args.p + 1

    # ------------------------------------------------------------------
    # Per-seed loop
    # ------------------------------------------------------------------
    all_weighted_r2 = []   # list of dicts, one per seed
    all_gd_gaps = []
    all_top3_gaps = []
    all_rankings = []
    per_seed_results = {}

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'='*70}")
        print(f"  SEED {seed}  ({seed_idx+1}/{len(args.seeds)})")
        print(f"{'='*70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ICLTransformer(
            input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, max_seq_len=args.n_support + 1,
        ).to(device)

        # Load or train
        ckpt_path = None
        if args.load_dir:
            candidate = os.path.join(args.load_dir, f'model_seed_{seed}.pt')
            if os.path.exists(candidate):
                ckpt_path = candidate

        if ckpt_path is not None:
            print(f"  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            if 'cfg' in ckpt:
                for key in ['n_support', 'p', 'noise', 'probe_lam']:
                    if key in ckpt['cfg']:
                        cfg[key] = ckpt['cfg'][key]
        else:
            print(f"  Training fresh model ({cfg['steps']} steps)...")
            t0 = time.time()
            losses = train_model(model, cfg, device)
            elapsed = time.time() - t0
            final_loss = float(np.mean(losses[-min(1000, len(losses)):]))
            print(f"  Training done: {elapsed/60:.1f} min, final loss={final_loss:.5f}")
            save_path = f'{args.out_dir}/model_seed_{seed}.pt'
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'seed': seed},
                       save_path)
            print(f"  Saved: {save_path}")

        # Per-layer readouts
        print(f"  Training readout heads...")
        t0 = time.time()
        readouts = train_layer_readouts(model, cfg, device, readout_steps=3000)
        print(f"  Readouts done: {time.time()-t0:.1f}s")

        # Algorithm identification
        print(f"  Running algorithm identification...")
        t0 = time.time()
        results = run_algorithm_identification(
            model, readouts, cfg, device, n_per_kappa=args.n_per_kappa)
        print(f"  Analysis done: {time.time()-t0:.1f}s")

        # Per-seed metrics
        wr2 = compute_weighted_r2(results)
        gd_gap, top3_gap, ranking = compute_gaps(wr2)
        all_weighted_r2.append(wr2)
        all_gd_gaps.append(gd_gap)
        all_top3_gaps.append(top3_gap)
        all_rankings.append(ranking)
        per_seed_results[seed] = {
            'weighted_r2': wr2, 'gd_gap': gd_gap,
            'top3_gap': top3_gap, 'ranking': ranking,
        }

        sorted_algos = sorted(wr2, key=lambda n: -wr2[n])
        print(f"  Weighted R²: " +
              ", ".join(f"{n}={wr2[n]:.4f}" for n in sorted_algos))
        print(f"  GD gap={gd_gap:.4f}  top3_gap={top3_gap:.4f}")
        print(f"  Ranking: {' > '.join(sorted_algos[:3])} > ...")

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    algo_names = list(ALGORITHMS.keys())
    n_seeds = len(args.seeds)

    mean_r2, std_r2 = {}, {}
    for name in algo_names:
        vals = [wr2[name] for wr2 in all_weighted_r2]
        mean_r2[name] = float(np.mean(vals))
        std_r2[name] = float(np.std(vals))

    mean_gd_gap = float(np.mean(all_gd_gaps))
    std_gd_gap = float(np.std(all_gd_gaps))
    mean_top3_gap = float(np.mean(all_top3_gaps))
    std_top3_gap = float(np.std(all_top3_gaps))

    top1_counts = Counter(r[0] for r in all_rankings)
    top1_most_common, top1_freq = top1_counts.most_common(1)[0]

    top3_tuples = [tuple(r[:3]) for r in all_rankings]
    top3_counts = Counter(top3_tuples)
    consensus_top3, consensus_top3_freq = top3_counts.most_common(1)[0]

    changed_seeds = [seed for seed, ranking in zip(args.seeds, all_rankings)
                     if tuple(ranking[:3]) != consensus_top3]

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("MULTI-SEED AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"\n  Kappa-weighted mean R² per algorithm (mean ± std across {n_seeds} seeds):")
    print(f"  {'Algorithm':>15}  {'Mean R²':>8}  {'Std':>8}")
    print(f"  {'-'*35}")
    for name in sorted(algo_names, key=lambda n: -mean_r2[n]):
        print(f"  {name:>15}  {mean_r2[name]:>8.4f}  {std_r2[name]:>8.4f}")

    print(f"\n  Model >> GD gap:  {mean_gd_gap:.4f} ± {std_gd_gap:.4f}")
    print(f"  Top-3 gap:        {mean_top3_gap:.4f} ± {std_top3_gap:.4f}")
    print(f"\n  Top-1 algorithm consistency: {top1_most_common} "
          f"in {top1_freq}/{n_seeds} seeds")
    print(f"  Consensus top-3: {' > '.join(consensus_top3)} "
          f"({consensus_top3_freq}/{n_seeds} seeds)")

    if changed_seeds:
        print(f"\n  Seeds with different top-3 ranking: {changed_seeds}")
        for seed in changed_seeds:
            r = per_seed_results[seed]['ranking']
            print(f"    seed {seed}: {' > '.join(r[:3])}")
    else:
        print(f"\n  All {n_seeds} seeds agree on the top-3 ranking.")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print(f"\n  Generating plots...")
    plot_ranking_bars(algo_names, mean_r2, std_r2, args.out_dir)
    plot_gap_per_seed(args.seeds, all_gd_gaps, all_top3_gaps, args.out_dir)
    print(f"  Saved: {args.out_dir}/multi_seed_ranking.png")
    print(f"  Saved: {args.out_dir}/multi_seed_gap.png")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {
        'seeds': args.seeds,
        'config': cfg,
        'per_seed': {
            str(seed): per_seed_results[seed] for seed in args.seeds
        },
        'aggregate': {
            'mean_weighted_r2': mean_r2,
            'std_weighted_r2': std_r2,
            'mean_gd_gap': mean_gd_gap,
            'std_gd_gap': std_gd_gap,
            'mean_top3_gap': mean_top3_gap,
            'std_top3_gap': std_top3_gap,
            'top1_most_common': top1_most_common,
            'top1_frequency': top1_freq,
            'consensus_top3': list(consensus_top3),
            'consensus_top3_frequency': consensus_top3_freq,
            'changed_seeds': changed_seeds,
        },
    }
    out_json = f'{args.out_dir}/multi_seed_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_json}")
    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
