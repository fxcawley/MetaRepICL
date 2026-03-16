"""Discrimination Stress Test: Can we separate CG-class algorithms at p >> L?

The key experiment for the workshop paper. At p=10-20 with L=12-24 layers,
all CG-class methods converge in <= p steps, making late layers uninformative
for discrimination. By scaling p >> L, every layer is mid-convergence and
algorithms should diverge maximally.

Sweep grid:
  p:          [10, 20, 40, 60, 80, 100]
  num_layers: [12, 24]
  kappas:     [10, 50, 100, 500, 1000]
  spectra:    [geometric, step, mixture]
  seeds:      [42, 123, 456]

Expected outcomes:
  A) At p >> L, one algorithm wins decisively -> "identification possible at scale"
  B) Top 3 remain clustered even at p=100 -> "indistinguishability is structural"
  Either is publishable.

Usage:
    python experiments/discrimination_sweep.py                    # full sweep
    python experiments/discrimination_sweep.py --quick            # quick test (2 p values)
    python experiments/discrimination_sweep.py --load-dir models/ # use pretrained models
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path
from itertools import product

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

# Import algorithm implementations from algorithm_id
from algorithm_id import (
    ALGORITHMS, ALGO_COLORS, KAPPAS as DEFAULT_KAPPAS,
    generate_batch_with_cov, generate_mixed_kappa_batch,
    train_model, train_layer_readouts, run_algorithm_identification,
)


# ---------------------------------------------------------------------------
# Extended data generation: different spectrum types
# ---------------------------------------------------------------------------

def generate_batch_step_spectrum(batch_size, n_support, p, cond, noise=0.1, device='cpu'):
    """Generate batch with STEP spectrum: eigenvalues are {1, cond}, half each.

    This creates a 'bimodal' conditioning that should make CG and PGD diverge
    more than geometric spacing (CG exploits spectral gaps better).
    """
    sigmas = torch.ones(p, device=device)
    sigmas[:p // 2] = cond  # half eigenvalues = cond, half = 1
    sqrt_sig = sigmas.sqrt()

    w = torch.randn(batch_size, p, 1, device=device) / math.sqrt(p)
    x_s = torch.randn(batch_size, n_support, p, device=device) * sqrt_sig
    y_s = (x_s @ w).squeeze(-1) + noise * torch.randn(batch_size, n_support, device=device)
    x_q = torch.randn(batch_size, 1, p, device=device) * sqrt_sig
    y_q = (x_q @ w).squeeze(-1).squeeze(-1)

    support_tokens = torch.cat([x_s, y_s.unsqueeze(-1)], dim=-1)
    query_token = torch.cat([x_q, torch.zeros(batch_size, 1, 1, device=device)], dim=-1)
    seq = torch.cat([support_tokens, query_token], dim=1)
    return seq, y_q, x_s, y_s, x_q


def generate_batch_mixture_kappa(batch_size, n_support, p, noise=0.1, device='cpu'):
    """Generate batch with MIXED condition numbers within the batch.

    Each problem gets a random kappa from {10, 500}, creating within-batch diversity.
    This is adversarial: algorithms that adapt to kappa should outperform fixed-step methods.
    """
    kappa = torch.where(
        torch.rand(batch_size, device=device) < 0.5,
        torch.tensor(10.0, device=device),
        torch.tensor(500.0, device=device),
    )
    # Per-problem generation
    seqs, y_qs, x_ss, y_ss, x_qs = [], [], [], [], []
    for i in range(batch_size):
        seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
            1, n_support, p, float(kappa[i]), noise, device)
        seqs.append(seq)
        y_qs.append(y_q)
        x_ss.append(x_s)
        y_ss.append(y_s)
        x_qs.append(x_q)
    return (torch.cat(seqs), torch.cat(y_qs),
            torch.cat(x_ss), torch.cat(y_ss), torch.cat(x_qs))


SPECTRUM_GENERATORS = {
    'geometric': generate_batch_with_cov,  # default: log-spaced eigenvalues
    'step': generate_batch_step_spectrum,   # bimodal: {1, kappa}
}

TRAINING_KAPPAS = [1.0, 10.0, 50.0, 100.0, 500.0]


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------

def run_single_config(p, num_layers, seeds, kappas, n_per_kappa, spectrum,
                      steps, device, out_dir, load_dir=None):
    """Train + evaluate one (p, num_layers) configuration across seeds.

    Returns dict with per-seed and aggregated results.
    """
    n_support = 2 * p  # ensure n > p
    input_dim = p + 1
    d_model = 256
    nhead = 4

    cfg = {
        'steps': steps, 'batch_size': 64, 'n_support': n_support, 'p': p,
        'noise': 0.1, 'lr': 3e-4, 'wd': 0.01,
        'warmup': min(1000, steps // 10),
        'log_every': max(1, steps // 25),
        'probe_lam': 0.1,
    }

    per_seed_results = {}
    for seed in seeds:
        print(f"\n    seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ICLTransformer(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, max_seq_len=n_support + 1,
        ).to(device)

        # Load or train
        ckpt_path = None
        if load_dir:
            ckpt_path = os.path.join(load_dir, f'model_p{p}_L{num_layers}_s{seed}.pt')
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"      Loading from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
        else:
            print(f"      Training ({steps} steps)...")
            t0 = time.time()
            train_model(model, cfg, device)
            print(f"      Trained in {(time.time()-t0)/60:.1f}m")
            # Save checkpoint
            save_dir = os.path.join(out_dir, 'checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {'model': model.state_dict(), 'cfg': cfg},
                os.path.join(save_dir, f'model_p{p}_L{num_layers}_s{seed}.pt')
            )

        # Train readouts
        readouts = train_layer_readouts(model, cfg, device, readout_steps=3000)

        # Run algorithm ID
        results = run_algorithm_identification(
            model, readouts, cfg, device, n_per_kappa=n_per_kappa)

        # Compute aggregate weighted R²
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
        best = sorted_algos[0]
        second = sorted_algos[1]
        third = sorted_algos[2]
        gap_1_2 = weighted_r2[best] - weighted_r2[second]
        gap_1_3 = weighted_r2[best] - weighted_r2[third]
        ci_1_2 = weighted_ci[best] + weighted_ci[second]
        gd_r2 = weighted_r2['GD']

        per_seed_results[seed] = {
            'weighted_r2': weighted_r2,
            'weighted_ci': weighted_ci,
            'ranking': sorted_algos,
            'best': best,
            'gap_1_2': gap_1_2,
            'gap_1_3': gap_1_3,
            'ci_1_2': ci_1_2,
            'gd_r2': gd_r2,
            'distinguishable': gap_1_2 > ci_1_2,
        }

    # Aggregate across seeds
    algo_names = list(ALGORITHMS.keys())
    agg = {}
    for name in algo_names:
        r2s = [per_seed_results[s]['weighted_r2'][name] for s in seeds]
        agg[name] = {'mean': float(np.mean(r2s)), 'std': float(np.std(r2s))}

    gaps_1_2 = [per_seed_results[s]['gap_1_2'] for s in seeds]
    gaps_1_3 = [per_seed_results[s]['gap_1_3'] for s in seeds]
    gd_gaps = [per_seed_results[s]['weighted_r2'][per_seed_results[s]['ranking'][0]]
               - per_seed_results[s]['gd_r2'] for s in seeds]

    return {
        'p': p, 'num_layers': num_layers, 'n_support': n_support,
        'per_seed': {int(s): per_seed_results[s] for s in seeds},
        'aggregate': agg,
        'gap_1_2': {'mean': float(np.mean(gaps_1_2)), 'std': float(np.std(gaps_1_2))},
        'gap_1_3': {'mean': float(np.mean(gaps_1_3)), 'std': float(np.std(gaps_1_3))},
        'gd_gap': {'mean': float(np.mean(gd_gaps)), 'std': float(np.std(gd_gaps))},
        'any_distinguishable': any(per_seed_results[s]['distinguishable'] for s in seeds),
        'all_distinguishable': all(per_seed_results[s]['distinguishable'] for s in seeds),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_discrimination_heatmap(all_results, out_dir):
    """Heatmap: gap between #1 and #2 algorithm as function of (p, L)."""
    p_vals = sorted(set(r['p'] for r in all_results))
    l_vals = sorted(set(r['num_layers'] for r in all_results))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gap heatmap
    gap_matrix = np.full((len(l_vals), len(p_vals)), np.nan)
    for r in all_results:
        i = l_vals.index(r['num_layers'])
        j = p_vals.index(r['p'])
        gap_matrix[i, j] = r['gap_1_2']['mean']

    ax = axes[0]
    im = ax.imshow(gap_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xticks(range(len(p_vals)))
    ax.set_xticklabels(p_vals)
    ax.set_yticks(range(len(l_vals)))
    ax.set_yticklabels(l_vals)
    ax.set_xlabel('Feature dimension p')
    ax.set_ylabel('Number of layers L')
    ax.set_title('Gap: Best vs 2nd-best algorithm\n(mean across seeds)')
    for i in range(len(l_vals)):
        for j in range(len(p_vals)):
            if not np.isnan(gap_matrix[i, j]):
                ax.text(j, i, f'{gap_matrix[i,j]:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # GD gap heatmap
    gd_matrix = np.full((len(l_vals), len(p_vals)), np.nan)
    for r in all_results:
        i = l_vals.index(r['num_layers'])
        j = p_vals.index(r['p'])
        gd_matrix[i, j] = r['gd_gap']['mean']

    ax = axes[1]
    im2 = ax.imshow(gd_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xticks(range(len(p_vals)))
    ax.set_xticklabels(p_vals)
    ax.set_yticks(range(len(l_vals)))
    ax.set_yticklabels(l_vals)
    ax.set_xlabel('Feature dimension p')
    ax.set_ylabel('Number of layers L')
    ax.set_title('Gap: Best algorithm vs GD\n(mean across seeds)')
    for i in range(len(l_vals)):
        for j in range(len(p_vals)):
            if not np.isnan(gd_matrix[i, j]):
                ax.text(j, i, f'{gd_matrix[i,j]:.3f}', ha='center', va='center',
                        fontsize=8, fontweight='bold')
    fig.colorbar(im2, ax=ax, shrink=0.8)

    plt.suptitle('Discrimination Sweep: Does p >> L Enable Algorithm Identification?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/discrimination_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_scaling_curves(all_results, out_dir):
    """Line plots: R² per algorithm as p increases, one panel per L."""
    l_vals = sorted(set(r['num_layers'] for r in all_results))
    algo_names = list(ALGORITHMS.keys())

    fig, axes = plt.subplots(1, len(l_vals), figsize=(7*len(l_vals), 5), squeeze=False)
    for idx, L in enumerate(l_vals):
        ax = axes[0, idx]
        configs = sorted([r for r in all_results if r['num_layers'] == L],
                         key=lambda r: r['p'])
        p_vals = [r['p'] for r in configs]

        for name in algo_names:
            means = [r['aggregate'][name]['mean'] for r in configs]
            stds = [r['aggregate'][name]['std'] for r in configs]
            ax.errorbar(p_vals, means, yerr=stds, marker='o', label=name,
                       color=ALGO_COLORS[name], linewidth=2, capsize=3)

        ax.set_xlabel('Feature dimension p')
        ax.set_ylabel('Weighted R²')
        ax.set_title(f'L = {L} layers')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Mark p = L
        ax.axvline(x=L, color='gray', linestyle='--', alpha=0.5, label=f'p = L = {L}')

    plt.suptitle('Algorithm R² vs Feature Dimension (p >> L = mid-convergence regime)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/discrimination_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_best_algorithm_map(all_results, out_dir):
    """Show which algorithm wins at each (p, L) configuration."""
    p_vals = sorted(set(r['p'] for r in all_results))
    l_vals = sorted(set(r['num_layers'] for r in all_results))
    algo_names = list(ALGORITHMS.keys())

    fig, ax = plt.subplots(figsize=(10, 4))

    for r in all_results:
        i = l_vals.index(r['num_layers'])
        j = p_vals.index(r['p'])
        # Best algorithm from aggregate
        best = max(r['aggregate'].keys(), key=lambda n: r['aggregate'][n]['mean'])
        color = ALGO_COLORS[best]
        marker = 's' if r['all_distinguishable'] else 'o'
        edge = 'black' if r['all_distinguishable'] else 'gray'
        ax.scatter(j, i, c=color, s=200, marker=marker, edgecolors=edge, linewidths=2)
        ax.text(j, i - 0.35, best, ha='center', va='top', fontsize=7)

    ax.set_xticks(range(len(p_vals)))
    ax.set_xticklabels(p_vals)
    ax.set_yticks(range(len(l_vals)))
    ax.set_yticklabels(l_vals)
    ax.set_xlabel('Feature dimension p')
    ax.set_ylabel('Number of layers L')
    ax.set_title('Best-Matching Algorithm by Configuration\n'
                 '(square = statistically distinguishable, circle = not)')

    # Legend for algorithm colors
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ALGO_COLORS[n], label=n) for n in algo_names]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/discrimination_best_algo.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Discrimination Sweep: p >> L stress test')
    parser.add_argument('--p-values', type=int, nargs='+', default=[10, 20, 40, 60, 80, 100],
                        help='Feature dimensions to sweep')
    parser.add_argument('--layer-values', type=int, nargs='+', default=[12, 24],
                        help='Number of layers to sweep')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Seeds for robustness')
    parser.add_argument('--kappas', type=float, nargs='+', default=[10.0, 50.0, 100.0, 500.0],
                        help='Condition numbers to test')
    parser.add_argument('--n-per-kappa', type=int, default=300,
                        help='Test problems per kappa')
    parser.add_argument('--steps', type=int, default=30000,
                        help='Training steps per model')
    parser.add_argument('--spectrum', type=str, default='geometric',
                        choices=['geometric', 'step'],
                        help='Eigenvalue spectrum type')
    parser.add_argument('--out-dir', type=str, default='docs/figures/discrimination_sweep')
    parser.add_argument('--load-dir', type=str, default=None,
                        help='Directory with pretrained model checkpoints')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 2 p values, 1 seed, fewer steps')
    args = parser.parse_args()

    if args.quick:
        args.p_values = [10, 40]
        args.layer_values = [12]
        args.seeds = [42]
        args.n_per_kappa = 50
        args.steps = 5000
        print("QUICK MODE: reduced sweep for testing")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Override KAPPAS used in training to match test kappas
    global TRAINING_KAPPAS
    TRAINING_KAPPAS = [1.0] + [k for k in args.kappas if k not in [1.0]]

    total_configs = len(args.p_values) * len(args.layer_values)
    print(f"\n{'='*70}")
    print(f"DISCRIMINATION SWEEP")
    print(f"  p values:    {args.p_values}")
    print(f"  L values:    {args.layer_values}")
    print(f"  seeds:       {args.seeds}")
    print(f"  kappas:      {args.kappas}")
    print(f"  spectrum:    {args.spectrum}")
    print(f"  configs:     {total_configs}")
    print(f"  total models to train: {total_configs * len(args.seeds)}")
    print(f"{'='*70}")

    all_results = []
    config_idx = 0
    t_total = time.time()

    for num_layers in args.layer_values:
        for p in args.p_values:
            config_idx += 1
            print(f"\n{'='*70}")
            print(f"  Config {config_idx}/{total_configs}: p={p}, L={num_layers}, "
                  f"n_support={2*p}, ratio p/L={p/num_layers:.1f}")
            print(f"{'='*70}")

            t0 = time.time()
            result = run_single_config(
                p=p, num_layers=num_layers, seeds=args.seeds,
                kappas=args.kappas, n_per_kappa=args.n_per_kappa,
                spectrum=args.spectrum, steps=args.steps,
                device=device, out_dir=args.out_dir,
                load_dir=args.load_dir,
            )
            elapsed = time.time() - t0

            all_results.append(result)

            # Print summary for this config
            print(f"\n  Summary (p={p}, L={num_layers}):")
            print(f"    Time: {elapsed/60:.1f} min")
            for name in sorted(result['aggregate'].keys(),
                             key=lambda n: -result['aggregate'][n]['mean']):
                r = result['aggregate'][name]
                print(f"    {name:>15}: {r['mean']:.4f} +/- {r['std']:.4f}")
            print(f"    Gap #1 vs #2: {result['gap_1_2']['mean']:.4f} "
                  f"+/- {result['gap_1_2']['std']:.4f}")
            print(f"    Gap vs GD:    {result['gd_gap']['mean']:.4f} "
                  f"+/- {result['gd_gap']['std']:.4f}")
            distinguishable = "YES" if result['all_distinguishable'] else "NO"
            print(f"    Distinguishable (all seeds): {distinguishable}")

    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE ({total_time/60:.1f} min total)")
    print(f"{'='*70}")

    # ---- Summary analysis ----
    print(f"\n{'='*70}")
    print("DISCRIMINATION ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  {'p':>4} {'L':>3} {'p/L':>5} | {'Best':>12} {'R2':>6} | "
          f"{'Gap12':>6} {'GapGD':>6} | Distinguishable?")
    print("  " + "-" * 75)
    for r in sorted(all_results, key=lambda x: (x['num_layers'], x['p'])):
        best = max(r['aggregate'].keys(), key=lambda n: r['aggregate'][n]['mean'])
        r2 = r['aggregate'][best]['mean']
        status = "YES" if r['all_distinguishable'] else "no"
        print(f"  {r['p']:4d} {r['num_layers']:3d} {r['p']/r['num_layers']:5.1f} | "
              f"{best:>12} {r2:.3f} | "
              f"{r['gap_1_2']['mean']:.4f} {r['gd_gap']['mean']:.4f} | {status}")

    # Check: does gap increase with p/L ratio?
    ratios = [r['p'] / r['num_layers'] for r in all_results]
    gaps = [r['gap_1_2']['mean'] for r in all_results]
    if len(ratios) > 2:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(ratios, gaps)
        print(f"\n  Spearman correlation (p/L ratio vs gap): rho={corr:.3f}, p={pval:.3f}")
        if corr > 0.3 and pval < 0.1:
            print("  => Gap INCREASES with p/L: larger p helps discrimination")
        elif corr < -0.3 and pval < 0.1:
            print("  => Gap DECREASES with p/L: indistinguishability deepens at scale")
        else:
            print("  => No clear trend: discrimination depends on factors beyond p/L ratio")

    # ---- Plots ----
    print(f"\n  Generating plots...")
    plot_discrimination_heatmap(all_results, args.out_dir)
    plot_scaling_curves(all_results, args.out_dir)
    plot_best_algorithm_map(all_results, args.out_dir)

    # ---- Save results ----
    # Convert numpy types for JSON serialization
    def to_json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]
        return obj

    save_data = {
        'sweep_config': {
            'p_values': args.p_values,
            'layer_values': args.layer_values,
            'seeds': args.seeds,
            'kappas': args.kappas,
            'spectrum': args.spectrum,
            'steps': args.steps,
            'n_per_kappa': args.n_per_kappa,
        },
        'results': to_json_safe(all_results),
        'total_time_min': total_time / 60,
    }
    with open(f'{args.out_dir}/discrimination_sweep_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
