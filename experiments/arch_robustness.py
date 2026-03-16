"""Architecture Robustness: Is algorithm identification stable across model configs?

Tests whether "Model >> GD" and "top-3 CG-class tie" hold across modest
variations in depth, width, and number of attention heads. This addresses
the reviewer concern that findings may be specific to one training recipe.

Configs tested (4 controlled variants):
  Base:  12 layers, d=256, 4 heads, p=10
  Deep:  24 layers, d=256, 4 heads, p=10
  Wide:  12 layers, d=512, 4 heads, p=10
  Heads: 12 layers, d=256, 8 heads, p=10

Usage:
    python experiments/arch_robustness.py                # full run
    python experiments/arch_robustness.py --quick         # 2 configs, fewer steps
    python experiments/arch_robustness.py --configs all   # all 4 configs
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.icl_transformer import ICLTransformer
from algorithm_id import (
    ALGORITHMS, ALGO_COLORS, KAPPAS,
    generate_mixed_kappa_batch,
    train_model, train_layer_readouts, run_algorithm_identification,
)

# ---------------------------------------------------------------------------
# Architecture configurations
# ---------------------------------------------------------------------------

ARCH_CONFIGS = {
    'base': {
        'num_layers': 12, 'd_model': 256, 'nhead': 4,
        'label': 'Base (12L, 256d, 4h)',
    },
    'deep': {
        'num_layers': 24, 'd_model': 256, 'nhead': 4,
        'label': 'Deep (24L, 256d, 4h)',
    },
    'wide': {
        'num_layers': 12, 'd_model': 512, 'nhead': 4,
        'label': 'Wide (12L, 512d, 4h)',
    },
    'heads': {
        'num_layers': 12, 'd_model': 256, 'nhead': 8,
        'label': 'More Heads (12L, 256d, 8h)',
    },
}


def run_arch_config(arch_name, arch, p, n_support, steps, seed,
                    n_per_kappa, device, out_dir, load_dir=None):
    """Train and evaluate one architecture configuration."""
    cfg = {
        'steps': steps, 'batch_size': 64, 'n_support': n_support, 'p': p,
        'noise': 0.1, 'lr': 3e-4, 'wd': 0.01,
        'warmup': min(1000, steps // 10),
        'log_every': max(1, steps // 25),
        'probe_lam': 0.1,
    }
    input_dim = p + 1

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ICLTransformer(
        input_dim=input_dim, d_model=arch['d_model'],
        nhead=arch['nhead'], num_layers=arch['num_layers'],
        max_seq_len=n_support + 1,
    ).to(device)
    n_params = sum(param.numel() for param in model.parameters())

    # Load or train
    ckpt_path = None
    if load_dir:
        ckpt_path = os.path.join(load_dir, f'model_{arch_name}_s{seed}.pt')
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"    Loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    else:
        print(f"    Training {arch['label']} ({n_params:,} params, {steps} steps)...")
        t0 = time.time()
        losses = train_model(model, cfg, device)
        train_time = time.time() - t0
        print(f"    Trained in {train_time/60:.1f}m, final loss: {np.mean(losses[-1000:]):.5f}")

        save_dir = os.path.join(out_dir, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {'model': model.state_dict(), 'cfg': cfg, 'arch': arch},
            os.path.join(save_dir, f'model_{arch_name}_s{seed}.pt')
        )

    # Train readouts
    readouts = train_layer_readouts(model, cfg, device, readout_steps=3000)

    # Algorithm ID
    results = run_algorithm_identification(
        model, readouts, cfg, device, n_per_kappa=n_per_kappa)

    # Aggregate weighted R²
    weighted_r2 = {}
    weighted_ci = {}
    for name in ALGORITHMS:
        total = sum(results[k]['mean_r2'][name] * math.log10(max(k, 1.1))
                    for k in results)
        total_ci = sum(results[k]['mean_r2_ci'][name] * math.log10(max(k, 1.1))
                      for k in results)
        norm = sum(math.log10(max(k, 1.1)) for k in results)
        weighted_r2[name] = total / norm
        weighted_ci[name] = total_ci / norm

    sorted_algos = sorted(weighted_r2.keys(), key=lambda n: -weighted_r2[n])
    gd_r2 = weighted_r2['GD']
    best_r2 = weighted_r2[sorted_algos[0]]

    return {
        'arch_name': arch_name,
        'arch': arch,
        'n_params': n_params,
        'weighted_r2': weighted_r2,
        'weighted_ci': weighted_ci,
        'ranking': sorted_algos,
        'gd_gap': best_r2 - gd_r2,
        'top3_gap': weighted_r2[sorted_algos[0]] - weighted_r2[sorted_algos[2]],
        'distinguishable': (weighted_r2[sorted_algos[0]] - weighted_r2[sorted_algos[1]]) > (
            weighted_ci[sorted_algos[0]] + weighted_ci[sorted_algos[1]]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_arch_comparison(all_results, out_dir):
    """Bar chart comparing algorithm R² across architectures."""
    algo_names = list(ALGORITHMS.keys())
    n_algos = len(algo_names)
    n_configs = len(all_results)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_configs)
    width = 0.12

    for j, name in enumerate(algo_names):
        means = [r['weighted_r2'][name] for r in all_results]
        cis = [r['weighted_ci'][name] for r in all_results]
        offset = (j - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=cis, label=name,
               color=ALGO_COLORS[name], alpha=0.85, capsize=2)

    labels = [r['arch']['label'] + f"\n({r['n_params']/1e6:.1f}M)" for r in all_results]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Kappa-Weighted R²')
    ax.set_title('Architecture Robustness: Algorithm Identification Across Model Configs',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/arch_robustness_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_arch_gaps(all_results, out_dir):
    """Show GD gap and top-3 gap across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [r['arch_name'] for r in all_results]
    labels = [r['arch']['label'] for r in all_results]

    # GD gap
    ax = axes[0]
    gd_gaps = [r['gd_gap'] for r in all_results]
    bars = ax.bar(range(len(names)), gd_gaps, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.set_ylabel('R² Gap (Best - GD)')
    ax.set_title('Gap: Best Algorithm vs GD\n(should be large and consistent)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(gd_gaps):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    # Top-3 gap
    ax = axes[1]
    top3_gaps = [r['top3_gap'] for r in all_results]
    bars = ax.bar(range(len(names)), top3_gaps, color='coral', alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.set_ylabel('R² Gap (1st - 3rd)')
    ax.set_title('Gap: Within CG-class (top 3)\n(should be small = indistinguishable)')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(top3_gaps):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/arch_robustness_gaps.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Architecture Robustness Sweep')
    parser.add_argument('--configs', type=str, nargs='+', default=['base', 'deep', 'wide', 'heads'],
                        choices=list(ARCH_CONFIGS.keys()) + ['all'],
                        help='Architecture configs to test')
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--n-support', type=int, default=20)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-per-kappa', type=int, default=200)
    parser.add_argument('--out-dir', type=str, default='docs/figures/arch_robustness')
    parser.add_argument('--load-dir', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if 'all' in args.configs:
        args.configs = list(ARCH_CONFIGS.keys())
    if args.quick:
        args.configs = ['base', 'deep']
        args.steps = 5000
        args.n_per_kappa = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("ARCHITECTURE ROBUSTNESS SWEEP")
    print(f"  Configs: {args.configs}")
    print(f"  p={args.p}, n_support={args.n_support}")
    print(f"{'='*60}")

    all_results = []
    for arch_name in args.configs:
        arch = ARCH_CONFIGS[arch_name]
        print(f"\n{'='*60}")
        print(f"  Config: {arch['label']}")
        print(f"{'='*60}")

        t0 = time.time()
        result = run_arch_config(
            arch_name, arch, args.p, args.n_support, args.steps,
            args.seed, args.n_per_kappa, device, args.out_dir, args.load_dir)
        elapsed = time.time() - t0

        all_results.append(result)

        print(f"\n  Results ({elapsed/60:.1f} min):")
        for name in result['ranking']:
            r2 = result['weighted_r2'][name]
            ci = result['weighted_ci'][name]
            print(f"    {name:>15}: {r2:.4f} +/- {ci:.4f}")
        print(f"  GD gap: {result['gd_gap']:.4f}")
        print(f"  Top-3 gap: {result['top3_gap']:.4f}")
        print(f"  Distinguishable: {'YES' if result['distinguishable'] else 'no'}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("ARCHITECTURE ROBUSTNESS SUMMARY")
    print(f"{'='*60}")

    print(f"\n  {'Config':>25} | {'Best':>12} | {'GD gap':>7} | {'Top3 gap':>8} | Stable?")
    print("  " + "-" * 75)
    for r in all_results:
        best = r['ranking'][0]
        stable = r['gd_gap'] > 0.10 and r['top3_gap'] < 0.05
        status = "YES" if stable else "MIXED"
        print(f"  {r['arch']['label']:>25} | {best:>12} | {r['gd_gap']:>7.4f} | "
              f"{r['top3_gap']:>8.4f} | {status}")

    # Check consistency
    all_gd_gaps = [r['gd_gap'] for r in all_results]
    all_top3_gaps = [r['top3_gap'] for r in all_results]
    gd_gap_stable = min(all_gd_gaps) > 0.05
    top3_stable = max(all_top3_gaps) < 0.10

    print(f"\n  GD gap range: [{min(all_gd_gaps):.4f}, {max(all_gd_gaps):.4f}]")
    print(f"    => {'STABLE: GD always clearly worse' if gd_gap_stable else 'UNSTABLE: GD gap varies'}")
    print(f"  Top-3 gap range: [{min(all_top3_gaps):.4f}, {max(all_top3_gaps):.4f}]")
    print(f"    => {'STABLE: Top 3 always clustered' if top3_stable else 'UNSTABLE: Some configs separate'}")

    # ---- Plots ----
    print(f"\n  Generating plots...")
    plot_arch_comparison(all_results, args.out_dir)
    plot_arch_gaps(all_results, args.out_dir)

    # ---- Save ----
    save_data = {
        'configs_tested': args.configs,
        'results': [{
            'arch_name': r['arch_name'],
            'arch': r['arch'],
            'n_params': r['n_params'],
            'weighted_r2': r['weighted_r2'],
            'weighted_ci': r['weighted_ci'],
            'ranking': r['ranking'],
            'gd_gap': r['gd_gap'],
            'top3_gap': r['top3_gap'],
            'distinguishable': r['distinguishable'],
        } for r in all_results],
        'gd_gap_stable': gd_gap_stable,
        'top3_stable': top3_stable,
    }
    with open(f'{args.out_dir}/arch_robustness_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
