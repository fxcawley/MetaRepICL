"""
Silent failure experiment: When does the softmax-KRR gap blow up?

This experiment tests the core MetaRep claim:
  "The softmax case introduces an exponential kernel that the transformer
   approximates but does not exactly implement. Understanding this gap ---
   when it is small and when it blows up --- is key to predicting where
   ICL will succeed and where it will fail silently."

We systematically measure the gap between:
  1. Exact exponential-kernel KRR (the oracle)
  2. Softmax attention (what the transformer actually computes)
  3. Linear attention CG (exact for dot-product kernels)

across four axes:
  - Temperature tau (controls kernel bandwidth)
  - Feature dimension p (high-d concentrates inner products)
  - Condition number kappa (ill-conditioning of the feature covariance)
  - Context length n (more support tokens => larger normalization Z)

"Silent failure" = the model produces plausible-looking predictions
(moderate RMSE) but is systematically wrong in a way that surface
metrics don't catch. We detect this via:
  - Per-query signed error (systematic bias, not just noise)
  - Rank correlation (predictions in wrong order)
  - Tail error (worst-case queries much worse than average)
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Core computation: exact KRR vs softmax vs linear attention
# ---------------------------------------------------------------------------

def make_features(n, p, seed, kappa=None):
    """Generate feature matrix, optionally with controlled condition number."""
    rng = np.random.default_rng(seed)
    if kappa is not None and kappa > 1:
        eigs = np.geomspace(1.0, float(kappa), p)
        Q, _ = np.linalg.qr(rng.standard_normal((n, p)).astype(np.float64))
        Q = Q[:n, :p]
        phi = Q * np.sqrt(eigs)[None, :]
    else:
        phi = rng.standard_normal((n, p)).astype(np.float64)
    return phi


def run_comparison(
    n_support=32, n_query=16, p=8, tau=1.0, lam=1e-2,
    noise=0.1, kappa=None, seed=42
):
    """
    Compare exact exp-kernel KRR, softmax attention, and linear-kernel CG.

    Returns dict with predictions and diagnostic metrics.
    """
    rng = np.random.default_rng(seed)

    # Features
    phi_all = make_features(n_support + n_query, p, seed, kappa=kappa)
    phi_s = phi_all[:n_support]
    phi_q = phi_all[n_support:]

    # Ground truth (linear function + noise)
    w_true = rng.standard_normal(p).astype(np.float64)
    y_s = phi_s @ w_true + noise * rng.standard_normal(n_support)
    y_q_true = phi_q @ w_true  # noiseless ground truth for queries

    # --- Exponential kernel ---
    S_ss = (phi_s @ phi_s.T) / tau          # (n, n)
    S_sq = (phi_s @ phi_q.T) / tau          # (n, nq)
    K_exp_ss = np.exp(S_ss)
    K_exp_sq = np.exp(S_sq)

    # Oracle KRR with exp kernel
    A = K_exp_ss + lam * np.eye(n_support)
    try:
        alpha_oracle = np.linalg.solve(A, y_s)
        f_oracle = K_exp_sq.T @ alpha_oracle
    except np.linalg.LinAlgError:
        f_oracle = np.full(n_query, np.nan)

    # --- Softmax attention (what the transformer computes) ---
    # Softmax normalizes each column of exp(S_sq) across support dimension
    # For predicting query j: w_ij = exp(S_ij) / sum_k exp(S_kj)
    # This is Nadaraya-Watson: f(x_q) = sum_i w_i y_i
    Z_sq = np.sum(np.exp(S_sq), axis=0, keepdims=True)  # (1, nq)
    W_softmax = np.exp(S_sq) / Z_sq                       # (n, nq)
    f_softmax = W_softmax.T @ y_s

    # --- Softmax with Z-recovery (aggregator-corrected) ---
    # If we can recover Z_i per row on support set, we can reconstruct K_exp
    # and solve KRR. This is what Route A claims a deep transformer does.
    Z_ss = np.sum(np.exp(S_ss), axis=1, keepdims=True)  # (n, 1)
    softmax_ss = np.exp(S_ss) / Z_ss
    K_recovered = softmax_ss * Z_ss                        # should == K_exp_ss
    A_recovered = K_recovered + lam * np.eye(n_support)
    try:
        alpha_recovered = np.linalg.solve(A_recovered, y_s)
        f_recovered = K_exp_sq.T @ alpha_recovered
    except np.linalg.LinAlgError:
        f_recovered = np.full(n_query, np.nan)

    # --- Linear kernel CG (Route B, exact) ---
    K_lin = phi_s @ phi_s.T
    A_lin = K_lin + lam * np.eye(n_support)
    try:
        alpha_lin = np.linalg.solve(A_lin, y_s)
        f_linear = (phi_s @ phi_q.T).T @ alpha_lin
    except np.linalg.LinAlgError:
        f_linear = np.full(n_query, np.nan)

    # --- Diagnostics ---
    def metrics(f_pred, label):
        if np.any(np.isnan(f_pred)):
            return {f"{label}_rmse": float('nan'), f"{label}_max_err": float('nan'),
                    f"{label}_bias": float('nan'), f"{label}_rank_corr": float('nan'),
                    f"{label}_tail90_err": float('nan')}
        err = f_pred - y_q_true
        abs_err = np.abs(err)
        rmse = float(np.sqrt(np.mean(err**2)))
        max_err = float(np.max(abs_err))
        bias = float(np.mean(err))  # systematic bias (signed)
        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(f_pred, y_q_true)
        # Tail error: mean error of worst 10% of queries
        tail_idx = abs_err >= np.percentile(abs_err, 90)
        tail_err = float(np.mean(abs_err[tail_idx]))
        return {
            f"{label}_rmse": rmse,
            f"{label}_max_err": max_err,
            f"{label}_bias": bias,
            f"{label}_rank_corr": float(rho) if not np.isnan(rho) else 0.0,
            f"{label}_tail90_err": tail_err,
        }

    # Normalization diagnostic: how big is Z?
    Z_mean = float(np.mean(Z_sq))
    Z_std = float(np.std(Z_sq))
    Z_max = float(np.max(Z_sq))
    Z_min = float(np.min(Z_sq))

    # Kernel condition number
    try:
        kappa_K = float(np.linalg.cond(K_exp_ss))
    except:
        kappa_K = float('nan')

    # Operator norm gap: ||K_recovered - K_exp||
    op_gap = float(np.linalg.norm(K_recovered - K_exp_ss, ord=2))

    result = {
        "tau": tau, "p": p, "n_support": n_support, "kappa_input": kappa,
        "lam": lam, "noise": noise, "seed": seed,
        "Z_mean": Z_mean, "Z_std": Z_std, "Z_max": Z_max, "Z_min": Z_min,
        "kappa_K_exp": kappa_K, "op_norm_gap": op_gap,
    }
    result.update(metrics(f_oracle, "oracle"))
    result.update(metrics(f_softmax, "softmax"))
    result.update(metrics(f_recovered, "recovered"))
    result.update(metrics(f_linear, "linear"))

    # Store predictions for plotting
    result["_f_oracle"] = f_oracle.tolist()
    result["_f_softmax"] = f_softmax.tolist()
    result["_f_recovered"] = f_recovered.tolist()
    result["_f_linear"] = f_linear.tolist()
    result["_y_true"] = y_q_true.tolist()

    return result


# ---------------------------------------------------------------------------
# Sweep experiments
# ---------------------------------------------------------------------------

def sweep_tau(seeds=range(42, 47)):
    """Sweep temperature: when does the softmax gap blow up?"""
    taus = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = []
    for tau in taus:
        for seed in seeds:
            r = run_comparison(n_support=32, n_query=16, p=8, tau=tau, seed=seed)
            results.append(r)
    return results


def sweep_dimension(seeds=range(42, 47)):
    """Sweep feature dimension: inner product concentration in high-d."""
    dims = [2, 4, 8, 16, 32, 64]
    results = []
    for p in dims:
        for seed in seeds:
            r = run_comparison(n_support=48, n_query=16, p=p, tau=1.0, seed=seed)
            results.append(r)
    return results


def sweep_kappa(seeds=range(42, 47)):
    """Sweep condition number: ill-conditioned features."""
    kappas = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    results = []
    for kappa in kappas:
        for seed in seeds:
            r = run_comparison(n_support=32, n_query=16, p=8, tau=1.0,
                               kappa=kappa, seed=seed)
            results.append(r)
    return results


def sweep_context(seeds=range(42, 47)):
    """Sweep context length: more tokens => larger Z."""
    ns = [8, 16, 32, 64, 128]
    results = []
    for n in ns:
        for seed in seeds:
            r = run_comparison(n_support=n, n_query=16, p=8, tau=1.0, seed=seed)
            results.append(r)
    return results


def silent_failure_demo(seed=42):
    """
    Construct a specific scenario where softmax attention gives plausible RMSE
    but is systematically wrong.

    Strategy: high tau + high dimension => inner products concentrate,
    softmax becomes nearly uniform, predictions collapse to mean(y).
    RMSE looks "ok" if y has low variance, but rank ordering is destroyed.
    """
    scenarios = {}

    # Scenario 1: Normal conditions (should work)
    scenarios["healthy"] = run_comparison(
        n_support=32, n_query=32, p=8, tau=1.0, noise=0.1, seed=seed
    )

    # Scenario 2: High-d concentration (silent failure)
    # In high dimensions, <phi_i, phi_j> concentrates around 0 for random features.
    # exp(0/tau) = 1 for all pairs => softmax becomes uniform => predictions = mean(y)
    scenarios["high_dim"] = run_comparison(
        n_support=32, n_query=32, p=64, tau=1.0, noise=0.1, seed=seed
    )

    # Scenario 3: Very low temperature (kernel matrix blows up)
    # Small tau => exp(large/tau) overflows or dominates => Z dominated by nearest neighbor
    # Prediction collapses to 1-NN instead of weighted regression
    scenarios["low_tau"] = run_comparison(
        n_support=32, n_query=32, p=8, tau=0.05, noise=0.1, seed=seed
    )

    # Scenario 4: Ill-conditioned features + moderate tau
    # Feature covariance is skewed => kernel has extreme eigenvalue spread
    # Softmax normalization washes out the informative directions
    scenarios["ill_cond"] = run_comparison(
        n_support=32, n_query=32, p=8, tau=0.5, noise=0.1, kappa=500.0, seed=seed
    )

    # Scenario 5: Long context (large n)
    # Z = sum of n exponentials grows with n
    # If features are random, Z ~ n * exp(||phi||^2 / (2*tau*p)) for self-similarity
    # Softmax weights become more uniform, less discriminative
    scenarios["long_context"] = run_comparison(
        n_support=128, n_query=32, p=8, tau=1.0, noise=0.1, seed=seed
    )

    return scenarios


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def aggregate_sweep(results, x_key):
    """Aggregate sweep results by x_key, computing mean/std over seeds."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r[x_key]].append(r)

    x_vals = sorted(groups.keys())
    agg = {x_key: x_vals}
    for metric in ["oracle_rmse", "softmax_rmse", "recovered_rmse", "linear_rmse",
                    "softmax_rank_corr", "softmax_tail90_err", "softmax_bias",
                    "Z_mean", "kappa_K_exp", "op_norm_gap"]:
        means = []
        stds = []
        for x in x_vals:
            vals = [r[metric] for r in groups[x] if not np.isnan(r.get(metric, float('nan')))]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(float('nan'))
                stds.append(0)
        agg[f"{metric}_mean"] = means
        agg[f"{metric}_std"] = stds
    return agg


def plot_sweep(agg, x_key, x_label, title, out_path):
    """Plot a 2x2 panel: RMSE, rank corr, tail error, Z magnitude."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = agg[x_key]

    # Panel 1: RMSE comparison
    ax = axes[0, 0]
    ax.plot(x, agg["oracle_rmse_mean"], 'b-o', label='Oracle (Exp KRR)', markersize=4)
    ax.plot(x, agg["softmax_rmse_mean"], 'r-s', label='Softmax Attention', markersize=4)
    ax.plot(x, agg["recovered_rmse_mean"], 'g-^', label='Z-Recovered KRR', markersize=4)
    ax.plot(x, agg["linear_rmse_mean"], 'k--d', label='Linear Kernel', markersize=3, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel('RMSE')
    ax.set_title('Prediction Error')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    if all(v > 0 for v in x):
        ax.set_xscale('log')

    # Panel 2: Rank correlation (does ordering survive?)
    ax = axes[0, 1]
    ax.plot(x, agg["softmax_rank_corr_mean"], 'r-s', label='Softmax', markersize=4)
    ax.axhline(1.0, color='blue', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Spearman Rank Correlation')
    ax.set_title('Rank Ordering Quality')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if all(v > 0 for v in x):
        ax.set_xscale('log')

    # Panel 3: Tail error (worst 10%)
    ax = axes[1, 0]
    ax.plot(x, agg["softmax_tail90_err_mean"], 'r-s', label='Softmax (P90 tail)', markersize=4)
    ax.plot(x, agg["softmax_rmse_mean"], 'r--', label='Softmax (avg)', alpha=0.5, markersize=3)
    ax.plot(x, agg["oracle_rmse_mean"], 'b--', label='Oracle (avg)', alpha=0.5, markersize=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Error')
    ax.set_title('Tail Error (Worst 10% of Queries)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    if all(v > 0 for v in x):
        ax.set_xscale('log')

    # Panel 4: Normalization magnitude Z
    ax = axes[1, 1]
    ax.plot(x, agg["Z_mean_mean"], 'purple', marker='o', markersize=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Mean Z (normalization constant)')
    ax.set_title('Softmax Normalization Magnitude')
    ax.grid(True, alpha=0.3)
    if all(v > 0 for v in x):
        ax.set_xscale('log')
    ax.set_yscale('log')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def plot_silent_failure(scenarios, out_path):
    """Show the silent failure cases side by side."""
    names = list(scenarios.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(4*n, 7))

    for i, name in enumerate(names):
        r = scenarios[name]
        y_true = np.array(r["_y_true"])
        f_oracle = np.array(r["_f_oracle"])
        f_softmax = np.array(r["_f_softmax"])

        idx = np.argsort(y_true)
        y_sorted = y_true[idx]

        # Top row: predictions vs ground truth
        ax = axes[0, i]
        ax.plot(y_sorted, y_sorted, 'k--', alpha=0.3, label='y=x')
        ax.scatter(y_sorted, f_oracle[idx], s=12, color='blue', alpha=0.6, label='Oracle')
        ax.scatter(y_sorted, f_softmax[idx], s=12, color='red', alpha=0.6, label='Softmax')
        ax.set_xlabel('True y')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{name}\nRMSE: oracle={r["oracle_rmse"]:.3f}, sm={r["softmax_rmse"]:.3f}',
                      fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        # Bottom row: signed error (bias detection)
        ax = axes[1, i]
        err_oracle = f_oracle[idx] - y_sorted
        err_softmax = f_softmax[idx] - y_sorted
        ax.bar(range(len(y_sorted)), err_softmax, color='red', alpha=0.5, label='Softmax err')
        ax.plot(range(len(y_sorted)), err_oracle, 'b-', alpha=0.5, linewidth=1, label='Oracle err')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Query (sorted by true y)')
        ax.set_ylabel('Signed Error')
        subtitle = (f'rank_corr={r["softmax_rank_corr"]:.2f}, '
                     f'bias={r["softmax_bias"]:.3f}')
        ax.set_title(subtitle, fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Silent Failure Detection: When Does Softmax ICL Break?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = "figures/silent_failure"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("SILENT FAILURE EXPERIMENT")
    print("Testing: When does the softmax-KRR gap blow up?")
    print("=" * 60)

    # 1. Temperature sweep
    print("\n[1/5] Sweeping temperature (tau)...")
    tau_results = sweep_tau()
    tau_agg = aggregate_sweep(tau_results, "tau")
    plot_sweep(tau_agg, "tau", "Temperature (tau)",
               "Temperature Sweep: Softmax-KRR Gap",
               f"{out_dir}/sweep_tau.png")

    # 2. Dimension sweep
    print("[2/5] Sweeping feature dimension (p)...")
    dim_results = sweep_dimension()
    dim_agg = aggregate_sweep(dim_results, "p")
    plot_sweep(dim_agg, "p", "Feature Dimension (p)",
               "Dimension Sweep: High-D Concentration Effect",
               f"{out_dir}/sweep_dimension.png")

    # 3. Condition number sweep
    print("[3/5] Sweeping condition number (kappa)...")
    kappa_results = sweep_kappa()
    kappa_agg = aggregate_sweep(kappa_results, "kappa_input")
    plot_sweep(kappa_agg, "kappa_input", "Condition Number (kappa)",
               "Conditioning Sweep: Ill-Conditioned Feature Covariance",
               f"{out_dir}/sweep_kappa.png")

    # 4. Context length sweep
    print("[4/5] Sweeping context length (n)...")
    context_results = sweep_context()
    context_agg = aggregate_sweep(context_results, "n_support")
    plot_sweep(context_agg, "n_support", "Context Length (n_support)",
               "Context Length Sweep: Z Growth",
               f"{out_dir}/sweep_context.png")

    # 5. Silent failure demo
    print("[5/5] Running silent failure scenarios...")
    scenarios = silent_failure_demo()
    plot_silent_failure(scenarios, f"{out_dir}/silent_failure_demo.png")

    # Print summary table
    print("\n" + "=" * 60)
    print("SILENT FAILURE SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<15} {'Oracle RMSE':>12} {'Softmax RMSE':>13} "
          f"{'Rank Corr':>10} {'Bias':>8} {'Tail Err':>9} {'Z_mean':>10}")
    print("-" * 80)
    for name, r in scenarios.items():
        print(f"{name:<15} {r['oracle_rmse']:>12.4f} {r['softmax_rmse']:>13.4f} "
              f"{r['softmax_rank_corr']:>10.3f} {r['softmax_bias']:>8.4f} "
              f"{r['softmax_tail90_err']:>9.4f} {r['Z_mean']:>10.1f}")

    print("\n--- Interpretation ---")
    # Detect silent failures: softmax RMSE looks ok but rank correlation is bad
    for name, r in scenarios.items():
        softmax_ok = r["softmax_rmse"] < 2 * r["oracle_rmse"] + 0.5
        rank_bad = r["softmax_rank_corr"] < 0.7
        if softmax_ok and rank_bad:
            print(f"  SILENT FAILURE: '{name}' -- RMSE looks acceptable "
                  f"({r['softmax_rmse']:.3f}) but rank ordering is destroyed "
                  f"(rho={r['softmax_rank_corr']:.3f})")
        elif not softmax_ok:
            print(f"  OVERT FAILURE: '{name}' -- RMSE clearly degraded "
                  f"({r['softmax_rmse']:.3f} vs oracle {r['oracle_rmse']:.3f})")
        else:
            print(f"  HEALTHY: '{name}' -- softmax tracks oracle well "
                  f"(rho={r['softmax_rank_corr']:.3f})")

    # Save JSON results
    json_out = {}
    for name, r in scenarios.items():
        json_out[name] = {k: v for k, v in r.items() if not k.startswith("_")}
    with open(f"{out_dir}/silent_failure_results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to {out_dir}/silent_failure_results.json")


if __name__ == "__main__":
    main()
