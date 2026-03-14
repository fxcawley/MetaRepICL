"""Algorithm Identification Experiment: What does the transformer actually learn?

Tests whether a trained ICL transformer matches a specific named iterative
algorithm. Instead of only asking "CG or GD?", we compare against a richer
set of algorithms that the model could plausibly implement:

  1. Vanilla GD (feature space)           -- baseline
  2. Vanilla CG (feature space)           -- existing baseline
  3. Preconditioned GD (Jacobi)           -- GD with M^{-1} = diag(X^T X + lam I)^{-1}
  4. Heavy Ball (Polyak momentum)         -- GD with momentum
  5. Chebyshev iteration                  -- optimal polynomial with predetermined steps
  6. Preconditioned CG (Jacobi)           -- CG with diagonal preconditioner

Key insight: Transformers can compute matrix-vector products (attention) but
struggle with scalar reductions like r^T r (needed for CG step sizes).
This makes preconditioned GD and Chebyshev iteration natural candidates --
they achieve CG-like convergence rates without requiring inner products.

The experiment measures:
  - Per-layer prediction MSE vs ground truth y_q (convergence profile match)
  - Per-problem prediction correlation (R^2 between model and algorithm)
  - Linear probe cosine similarity (state trajectory match)

All comparisons are stratified by condition number for maximum discrimination.

Usage:
    python experiments/algorithm_id.py                    # full run (loads trained model)
    python experiments/algorithm_id.py --steps 5000       # quick train + test
    python experiments/algorithm_id.py --load docs/figures/trained_mixed/model_mixed.pt
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

KAPPAS = [1.0, 10.0, 50.0, 100.0, 500.0]

# ---------------------------------------------------------------------------
# Data generation (same as train_mixed_kappa.py for consistency)
# ---------------------------------------------------------------------------

def generate_batch_with_cov(batch_size, n_support, p, cond, noise=0.1, device='cpu'):
    """Generate batch with controlled feature covariance condition number."""
    sigmas = torch.logspace(math.log10(max(cond, 1.01)), 0, p, device=device)
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


def generate_mixed_kappa_batch(batch_size, n_support, p, kappas, noise=0.1, device='cpu'):
    kappa = kappas[torch.randint(0, len(kappas), (1,)).item()]
    seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
        batch_size, n_support, p, kappa, noise, device)
    return seq, y_q


# ---------------------------------------------------------------------------
# Named iterative algorithms (all in FEATURE SPACE: solve (X^T X + lam I) w = X^T y)
# ---------------------------------------------------------------------------

def _feature_space_system(X, y, lam):
    """Build the normal equation system: B w = rhs where B = X^T X + lam I."""
    p = X.shape[1]
    B = X.T @ X + lam * np.eye(p)
    rhs = X.T @ y
    return B, rhs


def vanilla_gd_trajectory(X, y, lam, steps):
    """Gradient descent with optimal fixed step size on (X^T X + lam I) w = X^T y."""
    B, rhs = _feature_space_system(X, y, lam)
    eigs = np.linalg.eigvalsh(B)
    eta = 2.0 / (eigs[-1] + eigs[0])  # optimal step size
    w = np.zeros(X.shape[1])
    traj = [w.copy()]
    for _ in range(steps):
        grad = B @ w - rhs
        w = w - eta * grad
        traj.append(w.copy())
    return traj


def vanilla_cg_trajectory(X, y, lam, steps):
    """Conjugate Gradients on (X^T X + lam I) w = X^T y."""
    B, rhs = _feature_space_system(X, y, lam)
    p_dim = X.shape[1]
    w = np.zeros(p_dim)
    r = rhs.copy()
    d = r.copy()
    traj = [w.copy()]
    for _ in range(steps):
        Bd = B @ d
        rr = r @ r
        gamma = rr / (d @ Bd + 1e-30)
        w = w + gamma * d
        r = r - gamma * Bd
        beta = (r @ r) / (rr + 1e-30)
        d = r + beta * d
        traj.append(w.copy())
    return traj


def preconditioned_gd_trajectory(X, y, lam, steps):
    """Preconditioned GD with Jacobi (diagonal) preconditioner.

    Update: w_{t+1} = w_t - M^{-1} (B w_t - rhs)
    where M = diag(B) and step size eta is adapted for M^{-1} B.

    This is a natural fit for transformers: no inner products needed,
    just diagonal scaling (which can be stored in token embeddings).
    """
    B, rhs = _feature_space_system(X, y, lam)
    M_inv = 1.0 / np.diag(B)  # Jacobi preconditioner

    # Optimal step size for preconditioned system: 2 / (lam_max + lam_min) of M^{-1} B
    M_inv_B = np.diag(M_inv) @ B
    eigs_precond = np.linalg.eigvalsh(M_inv_B)
    eta = 2.0 / (eigs_precond[-1] + eigs_precond[0])

    w = np.zeros(X.shape[1])
    traj = [w.copy()]
    for _ in range(steps):
        grad = B @ w - rhs
        w = w - eta * (M_inv * grad)
        traj.append(w.copy())
    return traj


def heavy_ball_trajectory(X, y, lam, steps):
    """Polyak Heavy Ball method on (X^T X + lam I) w = X^T y.

    w_{t+1} = w_t - alpha * grad + beta * (w_t - w_{t-1})
    Optimal parameters: alpha = (2/(sqrt(L) + sqrt(mu)))^2
                        beta = ((sqrt(L) - sqrt(mu))/(sqrt(L) + sqrt(mu)))^2
    """
    B, rhs = _feature_space_system(X, y, lam)
    eigs = np.linalg.eigvalsh(B)
    L, mu = eigs[-1], eigs[0]
    sqrt_L, sqrt_mu = math.sqrt(L), math.sqrt(mu)

    alpha = (2.0 / (sqrt_L + sqrt_mu)) ** 2
    beta = ((sqrt_L - sqrt_mu) / (sqrt_L + sqrt_mu)) ** 2

    w = np.zeros(X.shape[1])
    w_prev = np.zeros(X.shape[1])
    traj = [w.copy()]
    for _ in range(steps):
        grad = B @ w - rhs
        w_new = w - alpha * grad + beta * (w - w_prev)
        w_prev = w
        w = w_new
        traj.append(w.copy())
    return traj


def chebyshev_iteration_trajectory(X, y, lam, steps):
    """Chebyshev semi-iterative method on (X^T X + lam I) w = X^T y.

    Uses predetermined step sizes based on eigenvalue bounds -- no inner
    products needed. Convergence rate matches CG in theory.

    Key property: step sizes omega_k are computed from Chebyshev polynomials
    using only the eigenvalue bounds (L, mu), NOT from the iterate.
    This makes it implementable by a transformer with fixed per-layer parameters.
    """
    B, rhs = _feature_space_system(X, y, lam)
    eigs = np.linalg.eigvalsh(B)
    L, mu = eigs[-1], eigs[0]

    # Chebyshev parameters
    c = (L + mu) / 2.0  # center
    e = (L - mu) / 2.0  # half-width

    w = np.zeros(X.shape[1])
    w_prev = np.zeros(X.shape[1])
    traj = [w.copy()]

    for k in range(steps):
        r = rhs - B @ w  # residual

        if k == 0:
            # First step: simple scaled gradient step
            omega = 1.0 / c
            w_new = w + omega * r
        else:
            # Chebyshev recursion parameters
            if k == 1:
                rho_prev = 1.0
            rho = 1.0 / (1.0 - (e / (2.0 * c)) ** 2 * rho_prev) if k == 1 else \
                  1.0 / (1.0 - (e ** 2 / (4.0 * c ** 2)) * rho_prev)
            omega = rho / c

            w_new = w + omega * r + (rho - 1.0) * (w - w_prev)
            rho_prev = rho

        w_prev = w
        w = w_new
        traj.append(w.copy())

    return traj


def preconditioned_cg_trajectory(X, y, lam, steps):
    """Preconditioned CG with Jacobi (diagonal) preconditioner.

    Standard PCG: replace r^T r with r^T M^{-1} r, use z = M^{-1} r.
    """
    B, rhs = _feature_space_system(X, y, lam)
    M_inv = 1.0 / np.diag(B)  # Jacobi preconditioner
    p_dim = X.shape[1]

    w = np.zeros(p_dim)
    r = rhs.copy()
    z = M_inv * r
    d = z.copy()
    rz = r @ z
    traj = [w.copy()]

    for _ in range(steps):
        Bd = B @ d
        dBd = d @ Bd
        gamma = rz / (dBd + 1e-30)
        w = w + gamma * d
        r = r - gamma * Bd
        z = M_inv * r
        rz_new = r @ z
        beta = rz_new / (rz + 1e-30)
        d = z + beta * d
        rz = rz_new
        traj.append(w.copy())

    return traj


ALGORITHMS = {
    'GD': vanilla_gd_trajectory,
    'CG': vanilla_cg_trajectory,
    'Precond GD': preconditioned_gd_trajectory,
    'Heavy Ball': heavy_ball_trajectory,
    'Chebyshev': chebyshev_iteration_trajectory,
    'Precond CG': preconditioned_cg_trajectory,
}

ALGO_COLORS = {
    'GD': 'red',
    'CG': 'green',
    'Precond GD': 'darkorange',
    'Heavy Ball': 'purple',
    'Chebyshev': 'brown',
    'Precond CG': 'teal',
}

# ---------------------------------------------------------------------------
# Training (reused from train_mixed_kappa.py)
# ---------------------------------------------------------------------------

def train_model(model, cfg, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['wd'], betas=(0.9, 0.98))
    warmup = cfg.get('warmup', 1000)
    total = cfg['steps']
    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    losses = []
    t0 = time.time()
    for step in range(total):
        seq, y_q = generate_mixed_kappa_batch(
            cfg['batch_size'], cfg['n_support'], cfg['p'], KAPPAS, cfg['noise'], device)
        pred = model(seq)
        loss = ((pred - y_q) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % cfg.get('log_every', 2000) == 0:
            avg = np.mean(losses[-cfg.get('log_every', 2000):])
            elapsed = time.time() - t0
            eta_min = (total - step - 1) / (step + 1) * elapsed / 60
            print(f"  step {step+1}/{total}  loss={avg:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta_min:.1f}m")
    return losses


def train_layer_readouts(model, cfg, device, readout_steps=3000):
    model.eval()
    num_layers = model.num_layers
    readouts = nn.ModuleList([
        nn.Linear(model.d_model, 1).to(device) for _ in range(num_layers)
    ])
    opt = torch.optim.Adam(readouts.parameters(), lr=1e-3)
    for step in range(readout_steps):
        seq, y_q = generate_mixed_kappa_batch(
            cfg['batch_size'], cfg['n_support'], cfg['p'], KAPPAS, cfg['noise'], device)
        with torch.no_grad():
            _, intermediates = model.forward_with_intermediates(seq)
        total_loss = 0.0
        for l in range(num_layers):
            h = intermediates[l + 1]
            pred = readouts[l](h).squeeze(-1)
            total_loss = total_loss + ((pred - y_q) ** 2).mean()
        opt.zero_grad()
        total_loss.backward()
        opt.step()
    return readouts


# ---------------------------------------------------------------------------
# Algorithm identification analysis
# ---------------------------------------------------------------------------

def run_algorithm_identification(model, readouts, cfg, device, n_per_kappa=200):
    """Compare model per-layer predictions against all named algorithms.

    For each (kappa, problem, layer), we have:
      - model prediction (from readout head)
      - 6 algorithm predictions (from running each algorithm for that many steps)
    
    We measure:
      1. MSE of each method vs ground truth y_q (convergence profile)
      2. Per-problem correlation: R^2 between model and each algorithm's predictions
      3. "Best match" at each layer: which algorithm's MSE curve most closely
         tracks the model's MSE curve?
    """
    model.eval()
    num_layers = model.num_layers
    n_support = cfg['n_support']
    p_dim = cfg['p']
    lam = cfg.get('probe_lam', 0.1)

    results = {}
    for kappa in KAPPAS:
        print(f"  Analyzing kappa={kappa:.0f}...")

        per_layer_model_mse = [[] for _ in range(num_layers)]
        per_layer_algo_mse = {name: [[] for _ in range(num_layers)] for name in ALGORITHMS}
        per_layer_correlations = {name: [[] for _ in range(num_layers)] for name in ALGORITHMS}

        # Also collect per-problem predictions for correlation analysis
        per_layer_model_preds = [[] for _ in range(num_layers)]
        per_layer_algo_preds = {name: [[] for _ in range(num_layers)] for name in ALGORITHMS}
        per_layer_y_true = [[] for _ in range(num_layers)]

        actual_kappa_feat = []

        bs = 50
        for start in range(0, n_per_kappa, bs):
            cur_bs = min(bs, n_per_kappa - start)
            seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
                cur_bs, n_support, p_dim, kappa, cfg['noise'], device)

            with torch.no_grad():
                _, intermediates = model.forward_with_intermediates(seq)

            x_s_np = x_s.cpu().numpy()
            y_s_np = y_s.cpu().numpy()
            x_q_np = x_q.squeeze(1).cpu().numpy()
            y_q_np = y_q.cpu().numpy()

            for i in range(cur_bs):
                X_i = x_s_np[i]
                y_vec = y_s_np[i]
                xq_i = x_q_np[i]
                yq_i = y_q_np[i]

                # Condition number of feature-space system
                B = X_i.T @ X_i + lam * np.eye(p_dim)
                eigs = np.linalg.eigvalsh(B)
                actual_kappa_feat.append(float(eigs[-1] / max(eigs[0], 1e-30)))

                # Run all algorithms
                trajs = {}
                for name, algo_fn in ALGORITHMS.items():
                    try:
                        trajs[name] = algo_fn(X_i, y_vec, lam, num_layers)
                    except Exception:
                        # Some algorithms may fail on degenerate systems
                        trajs[name] = [np.zeros(p_dim)] * (num_layers + 1)

                for l in range(num_layers):
                    # Model prediction at layer l
                    h_l = intermediates[l + 1][i:i+1]
                    with torch.no_grad():
                        pred_l = readouts[l](h_l).item()

                    per_layer_model_mse[l].append((pred_l - yq_i) ** 2)
                    per_layer_model_preds[l].append(pred_l)
                    per_layer_y_true[l].append(yq_i)

                    # Algorithm predictions at step l+1
                    for name in ALGORITHMS:
                        w_algo = trajs[name][l + 1]
                        f_algo = float(xq_i @ w_algo)
                        per_layer_algo_mse[name][l].append((f_algo - yq_i) ** 2)
                        per_layer_algo_preds[name][l].append(f_algo)

        # Aggregate: MSE per layer
        model_mse = [float(np.mean(e)) for e in per_layer_model_mse]
        algo_mses = {name: [float(np.mean(e)) for e in per_layer_algo_mse[name]]
                     for name in ALGORITHMS}

        # Compute per-layer R^2 between model predictions and each algorithm's predictions
        algo_r2 = {}
        for name in ALGORITHMS:
            r2_by_layer = []
            for l in range(num_layers):
                m_preds = np.array(per_layer_model_preds[l])
                a_preds = np.array(per_layer_algo_preds[name][l])
                # R^2: how well does the algorithm predict the model's output?
                ss_res = np.sum((m_preds - a_preds) ** 2)
                ss_tot = np.sum((m_preds - np.mean(m_preds)) ** 2)
                if ss_tot < 1e-12:
                    r2 = 1.0 if ss_res < 1e-12 else 0.0
                else:
                    r2 = max(0.0, 1.0 - ss_res / ss_tot)
                r2_by_layer.append(float(r2))
            algo_r2[name] = r2_by_layer

        # Identify best-matching algorithm per layer
        best_match = []
        for l in range(num_layers):
            best_name = max(ALGORITHMS.keys(), key=lambda n: algo_r2[n][l])
            best_match.append(best_name)

        # Overall best match: which algorithm has highest mean R^2 across layers?
        mean_r2 = {name: float(np.mean(algo_r2[name])) for name in ALGORITHMS}
        overall_best = max(mean_r2.keys(), key=lambda n: mean_r2[n])

        # MSE profile distance: how close is the model's convergence curve to each algorithm?
        model_mse_arr = np.array(model_mse)
        mse_profile_dist = {}
        for name in ALGORITHMS:
            algo_mse_arr = np.array(algo_mses[name])
            # Normalized L2 distance between log-MSE curves
            log_model = np.log10(model_mse_arr + 1e-12)
            log_algo = np.log10(algo_mse_arr + 1e-12)
            dist = float(np.sqrt(np.mean((log_model - log_algo) ** 2)))
            mse_profile_dist[name] = dist
        profile_best = min(mse_profile_dist.keys(), key=lambda n: mse_profile_dist[n])

        results[kappa] = {
            'model_mse': model_mse,
            'algo_mses': algo_mses,
            'algo_r2': algo_r2,
            'mean_r2': mean_r2,
            'overall_best': overall_best,
            'mse_profile_dist': mse_profile_dist,
            'profile_best': profile_best,
            'best_match_per_layer': best_match,
            'actual_kappa_feat': float(np.mean(actual_kappa_feat)),
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence_comparison(results, out_dir):
    """Plot MSE convergence: model vs all algorithms, one panel per kappa."""
    kappas = sorted(results.keys())
    fig, axes = plt.subplots(1, len(kappas), figsize=(4.5*len(kappas), 4.5), squeeze=False)
    for i, kappa in enumerate(kappas):
        ax = axes[0, i]
        d = results[kappa]
        layers = list(range(1, len(d['model_mse']) + 1))

        # Model
        ax.semilogy(layers, d['model_mse'], '-o', color='blue', markersize=5,
                     linewidth=2.5, label='Trained Model', zorder=10)

        # All algorithms
        for name in ALGORITHMS:
            ax.semilogy(layers, d['algo_mses'][name], '--', color=ALGO_COLORS[name],
                         linewidth=1.5, label=name, alpha=0.8)

        kf = d.get('actual_kappa_feat', kappa)
        ax.set_xlabel('Layer / Step')
        ax.set_ylabel('MSE vs y_q')
        ax.set_title(f'kappa_in={kappa:.0f}  cond_feat={kf:.0f}\n'
                     f'Best profile match: {d["profile_best"]}',
                     fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Algorithm Identification: Model Convergence vs Named Algorithms\n'
                 '(all in feature space, measured against ground truth y_q)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/algo_id_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_r2_heatmap(results, out_dir):
    """Heatmap: R^2 between model predictions and each algorithm, per layer and kappa."""
    kappas = sorted(results.keys())
    algo_names = list(ALGORITHMS.keys())

    fig, axes = plt.subplots(1, len(kappas), figsize=(4*len(kappas), 3.5), squeeze=False)
    for i, kappa in enumerate(kappas):
        ax = axes[0, i]
        d = results[kappa]
        num_layers = len(d['model_mse'])

        # Build matrix: rows = algorithms, cols = layers
        matrix = np.zeros((len(algo_names), num_layers))
        for j, name in enumerate(algo_names):
            matrix[j] = d['algo_r2'][name]

        im = ax.imshow(matrix, aspect='auto', vmin=0, vmax=1, cmap='YlOrRd',
                       interpolation='nearest')
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels(range(1, num_layers + 1), fontsize=7)
        ax.set_yticks(range(len(algo_names)))
        ax.set_yticklabels(algo_names, fontsize=8)
        ax.set_xlabel('Layer', fontsize=8)
        ax.set_title(f'kappa={kappa:.0f}\nBest: {d["overall_best"]} '
                     f'(R^2={d["mean_r2"][d["overall_best"]]:.3f})', fontsize=9)

        # Add text annotations
        for j in range(len(algo_names)):
            for l in range(num_layers):
                val = matrix[j, l]
                color = 'white' if val > 0.5 else 'black'
                ax.text(l, j, f'{val:.2f}', ha='center', va='center',
                        fontsize=5, color=color)

    fig.colorbar(im, ax=axes[0, -1], shrink=0.8, label='R^2')
    plt.suptitle('Per-Problem Prediction R^2: Model vs Each Algorithm',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/algo_id_r2_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_bars(results, out_dir):
    """Bar chart: mean R^2 across layers for each algorithm, grouped by kappa."""
    kappas = sorted(results.keys())
    algo_names = list(ALGORITHMS.keys())
    n_algos = len(algo_names)
    n_kappas = len(kappas)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_kappas)
    width = 0.12

    for j, name in enumerate(algo_names):
        means = [results[k]['mean_r2'][name] for k in kappas]
        offset = (j - n_algos / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=name,
                      color=ALGO_COLORS[name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'kappa={k:.0f}' for k in kappas])
    ax.set_ylabel('Mean R^2 (model vs algorithm)')
    ax.set_title('Algorithm Identification Summary:\n'
                 'Which algorithm best predicts the transformer\'s per-problem outputs?',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/algo_id_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_mse_profile_distance(results, out_dir):
    """Bar chart: MSE profile distance (log-scale L2) for each algorithm by kappa."""
    kappas = sorted(results.keys())
    algo_names = list(ALGORITHMS.keys())
    n_algos = len(algo_names)
    n_kappas = len(kappas)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_kappas)
    width = 0.12

    for j, name in enumerate(algo_names):
        dists = [results[k]['mse_profile_dist'][name] for k in kappas]
        offset = (j - n_algos / 2 + 0.5) * width
        ax.bar(x + offset, dists, width, label=name,
               color=ALGO_COLORS[name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'kappa={k:.0f}' for k in kappas])
    ax.set_ylabel('MSE Profile Distance (lower = closer match)')
    ax.set_title('Convergence Profile Match:\n'
                 'Distance between model and algorithm MSE curves (log scale)',
                 fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/algo_id_profile_distance.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Algorithm Identification Experiment')
    parser.add_argument('--steps', type=int, default=50000, help='Training steps (if not loading)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-support', type=int, default=20)
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='docs/figures/algo_id')
    parser.add_argument('--n-per-kappa', type=int, default=200)
    parser.add_argument('--load', type=str, default=None, help='Path to pretrained model checkpoint')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = {
        'steps': args.steps, 'batch_size': args.batch_size,
        'n_support': args.n_support, 'p': args.p,
        'noise': args.noise, 'lr': args.lr, 'wd': 0.01,
        'warmup': min(1000, args.steps // 10),
        'log_every': max(1, args.steps // 25),
        'probe_lam': 0.1,
    }
    input_dim = args.p + 1

    # ---- Phase 1: Load or Train model ----
    model = ICLTransformer(
        input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, max_seq_len=args.n_support + 1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.load and os.path.exists(args.load):
        print(f"\n{'='*60}")
        print(f"Loading pretrained model from {args.load}")
        print(f"{'='*60}")
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'cfg' in ckpt:
            # Inherit training config for data generation consistency
            for key in ['n_support', 'p', 'noise', 'probe_lam']:
                if key in ckpt['cfg']:
                    cfg[key] = ckpt['cfg'][key]
        print(f"  Parameters: {n_params:,}")
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: Training ICL Transformer on mixed kappa {KAPPAS}")
        print(f"  {args.num_layers} layers, d={args.d_model}, {args.steps} steps")
        print(f"{'='*60}")
        print(f"  Parameters: {n_params:,}")
        t0 = time.time()
        losses = train_model(model, cfg, device)
        train_time = time.time() - t0
        print(f"\n  Training: {train_time/60:.1f} min, final loss: {np.mean(losses[-1000:]):.5f}")
        torch.save({'model': model.state_dict(), 'cfg': cfg}, f'{args.out_dir}/model_algo_id.pt')

    # ---- Phase 2: Train per-layer readouts ----
    print(f"\n{'='*60}")
    print("Phase 2: Training per-layer readout heads")
    print(f"{'='*60}")
    t0 = time.time()
    readouts = train_layer_readouts(model, cfg, device, readout_steps=3000)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Phase 3: Algorithm identification ----
    print(f"\n{'='*60}")
    print("Phase 3: Algorithm Identification Analysis")
    print(f"  Comparing model against: {', '.join(ALGORITHMS.keys())}")
    print(f"  Kappas: {KAPPAS}")
    print(f"  {args.n_per_kappa} problems per kappa")
    print(f"{'='*60}")

    t0 = time.time()
    results = run_algorithm_identification(
        model, readouts, cfg, device, n_per_kappa=args.n_per_kappa)
    print(f"  Total analysis time: {time.time()-t0:.1f}s")

    # ---- Print results ----
    print(f"\n{'='*60}")
    print("RESULTS: Algorithm Identification")
    print(f"{'='*60}")

    print(f"\n  Per-problem R^2 (model predictions vs algorithm predictions):")
    print(f"  {'kappa':>6} | " + " | ".join(f"{n:>11}" for n in ALGORITHMS) + " | Best")
    print("  " + "-" * (8 + 14 * len(ALGORITHMS) + 8))
    for kappa in sorted(results.keys()):
        d = results[kappa]
        vals = " | ".join(f"{d['mean_r2'][n]:>11.4f}" for n in ALGORITHMS)
        print(f"  {kappa:6.0f} | {vals} | {d['overall_best']}")

    print(f"\n  MSE profile distance (lower = model's convergence curve matches better):")
    print(f"  {'kappa':>6} | " + " | ".join(f"{n:>11}" for n in ALGORITHMS) + " | Best")
    print("  " + "-" * (8 + 14 * len(ALGORITHMS) + 8))
    for kappa in sorted(results.keys()):
        d = results[kappa]
        vals = " | ".join(f"{d['mse_profile_dist'][n]:>11.4f}" for n in ALGORITHMS)
        print(f"  {kappa:6.0f} | {vals} | {d['profile_best']}")

    # ---- Overall verdict ----
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    # Aggregate across kappas (weighted by kappa to emphasize harder problems)
    weighted_r2 = {}
    for name in ALGORITHMS:
        total = sum(results[k]['mean_r2'][name] * math.log10(max(k, 1.1))
                    for k in results)
        norm = sum(math.log10(max(k, 1.1)) for k in results)
        weighted_r2[name] = total / norm

    print(f"\n  Kappa-weighted mean R^2 (emphasizing harder problems):")
    for name, r2 in sorted(weighted_r2.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if r2 == max(weighted_r2.values()) else ""
        print(f"    {name:>15}: {r2:.4f}{marker}")

    best_algo = max(weighted_r2.keys(), key=lambda n: weighted_r2[n])
    second_best = sorted(weighted_r2.keys(), key=lambda n: -weighted_r2[n])[1]
    gap = weighted_r2[best_algo] - weighted_r2[second_best]

    if gap > 0.02:
        print(f"\n  CONCLUSION: The transformer most closely implements {best_algo}")
        print(f"    (R^2 advantage of {gap:.3f} over {second_best})")
    elif gap > 0.005:
        print(f"\n  CONCLUSION: The transformer is marginally closest to {best_algo}")
        print(f"    but {second_best} is also close (gap={gap:.3f})")
    else:
        print(f"\n  CONCLUSION: {best_algo} and {second_best} are indistinguishable")
        print(f"    (gap={gap:.4f})")

    # ---- Phase 4: Plots ----
    print(f"\n  Generating plots...")
    plot_convergence_comparison(results, args.out_dir)
    plot_r2_heatmap(results, args.out_dir)
    plot_summary_bars(results, args.out_dir)
    plot_mse_profile_distance(results, args.out_dir)
    print(f"  Plots saved to {args.out_dir}/")

    # ---- Save results ----
    save_results = {
        'config': cfg,
        'algorithms': list(ALGORITHMS.keys()),
        'weighted_r2': weighted_r2,
        'best_algorithm': best_algo,
    }
    for kappa in sorted(results.keys()):
        d = results[kappa]
        save_results[f'kappa_{kappa:.0f}'] = {
            'model_mse': d['model_mse'],
            'algo_mses': d['algo_mses'],
            'algo_r2': d['algo_r2'],
            'mean_r2': d['mean_r2'],
            'overall_best': d['overall_best'],
            'mse_profile_dist': d['mse_profile_dist'],
            'profile_best': d['profile_best'],
            'best_match_per_layer': d['best_match_per_layer'],
            'actual_kappa_feat': d['actual_kappa_feat'],
        }
    with open(f'{args.out_dir}/algo_id_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Results saved to {args.out_dir}/algo_id_results.json")


if __name__ == '__main__':
    main()
