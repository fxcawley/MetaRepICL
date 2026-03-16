"""Metric Robustness: Multi-metric agreement check for algorithm identification.

Extends algorithm_id.py with five additional metrics to verify that conclusions
about the transformer implementing a CG-class algorithm are robust:
  1. RMSE on predictions       -- sqrt(mean((model_pred - algo_pred)^2)) per layer
  2. Spearman rank correlation -- rank corr of per-problem squared errors
  3. Improvement correlation   -- Pearson corr of error reduction l -> l+1
  4. Pairwise wins             -- fraction of problems where algo is closest
  5. Final-iterate param error -- ||w_effective - w_algo|| at final layer

For each metric, reports whether (a) GD is clearly worse than CG class,
(b) the top 3 (CG, PCG, PGD) are clustered.

Usage:
    python experiments/metric_robustness.py --load path/to/model.pt
    python experiments/metric_robustness.py --load model.pt --results-dir docs/figures/algo_id
    python experiments/metric_robustness.py --steps 5000 --n-per-kappa 50
"""

import sys, os, json, time, math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.models.icl_transformer import ICLTransformer

KAPPAS = [1.0, 10.0, 50.0, 100.0, 500.0]

# -- Data generation (identical to algorithm_id.py) --------------------------

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

# -- Named iterative algorithms (duplicated for self-containment) ------------

def _feature_space_system(X, y, lam):
    """Build normal equation system: B w = rhs where B = X^T X + lam I."""
    p = X.shape[1]
    B = X.T @ X + lam * np.eye(p)
    return B, X.T @ y

def vanilla_gd_trajectory(X, y, lam, steps):
    B, rhs = _feature_space_system(X, y, lam)
    eigs = np.linalg.eigvalsh(B)
    eta = 2.0 / (eigs[-1] + eigs[0])
    w = np.zeros(X.shape[1]); traj = [w.copy()]
    for _ in range(steps):
        w = w - eta * (B @ w - rhs); traj.append(w.copy())
    return traj

def vanilla_cg_trajectory(X, y, lam, steps):
    B, rhs = _feature_space_system(X, y, lam)
    w = np.zeros(X.shape[1]); r = rhs.copy(); d = r.copy(); traj = [w.copy()]
    for _ in range(steps):
        Bd = B @ d; rr = r @ r
        gamma = rr / (d @ Bd + 1e-30)
        w = w + gamma * d; r = r - gamma * Bd
        beta = (r @ r) / (rr + 1e-30); d = r + beta * d
        traj.append(w.copy())
    return traj

def preconditioned_gd_trajectory(X, y, lam, steps):
    B, rhs = _feature_space_system(X, y, lam)
    M_inv = 1.0 / np.diag(B)
    M_inv_half = np.sqrt(M_inv)
    symm = np.diag(M_inv_half) @ B @ np.diag(M_inv_half)
    eigs = np.linalg.eigvalsh(symm)
    eta = 2.0 / (eigs[-1] + eigs[0])
    w = np.zeros(X.shape[1]); traj = [w.copy()]
    for _ in range(steps):
        w = w - eta * (M_inv * (B @ w - rhs)); traj.append(w.copy())
    return traj

def heavy_ball_trajectory(X, y, lam, steps):
    B, rhs = _feature_space_system(X, y, lam)
    eigs = np.linalg.eigvalsh(B)
    L, mu = eigs[-1], eigs[0]
    sqrt_L, sqrt_mu = math.sqrt(L), math.sqrt(mu)
    beta_opt = ((sqrt_L - sqrt_mu) / (sqrt_L + sqrt_mu)) ** 2
    alpha_opt = (2.0 / (sqrt_L + sqrt_mu)) ** 2
    beta = min(beta_opt, 0.9)
    alpha = min(alpha_opt, 2.0 * (1.0 + beta) / L * 0.95)
    w = np.zeros(X.shape[1]); w_prev = np.zeros(X.shape[1]); traj = [w.copy()]
    for _ in range(steps):
        w_new = w - alpha * (B @ w - rhs) + beta * (w - w_prev)
        w_prev = w; w = w_new; traj.append(w.copy())
    return traj

def chebyshev_iteration_trajectory(X, y, lam, steps):
    B, rhs = _feature_space_system(X, y, lam)
    eigs = np.linalg.eigvalsh(B)
    L, mu = eigs[-1], eigs[0]
    c, d_half = (L + mu) / 2.0, (L - mu) / 2.0
    sigma = d_half / c
    w = np.zeros(X.shape[1]); w_prev = w.copy()
    r0_norm = np.linalg.norm(rhs); traj = [w.copy()]; rho = None
    for k in range(steps):
        r = rhs - B @ w
        if k == 0:
            omega = 1.0 / c; w_new = w + omega * r; rho = 1.0
        else:
            rho_new = 1.0 / (1.0 - (sigma ** 2 / 4.0) * rho)
            omega = rho_new / c
            w_new = w + omega * r + (rho_new - 1.0) * (w - w_prev); rho = rho_new
        if np.linalg.norm(w_new) > 100.0 * (r0_norm + 1.0):
            w_new = w.copy()
        w_prev = w; w = w_new; traj.append(w.copy())
    return traj

def preconditioned_cg_trajectory(X, y, lam, steps):
    B, rhs = _feature_space_system(X, y, lam)
    M_inv = 1.0 / np.diag(B)
    w = np.zeros(X.shape[1]); r = rhs.copy(); z = M_inv * r
    d = z.copy(); rz = r @ z; traj = [w.copy()]
    for _ in range(steps):
        Bd = B @ d; gamma = rz / (d @ Bd + 1e-30)
        w = w + gamma * d; r = r - gamma * Bd; z = M_inv * r
        rz_new = r @ z; beta = rz_new / (rz + 1e-30)
        d = z + beta * d; rz = rz_new; traj.append(w.copy())
    return traj

ALGORITHMS = {
    'GD': vanilla_gd_trajectory, 'CG': vanilla_cg_trajectory,
    'Precond GD': preconditioned_gd_trajectory,
    'Heavy Ball': heavy_ball_trajectory,
    'Chebyshev': chebyshev_iteration_trajectory,
    'Precond CG': preconditioned_cg_trajectory,
}
ALGO_COLORS = {
    'GD': 'red', 'CG': 'green', 'Precond GD': 'darkorange',
    'Heavy Ball': 'purple', 'Chebyshev': 'brown', 'Precond CG': 'teal',
}

# -- Training helpers (identical to algorithm_id.py) -------------------------

def train_model(model, cfg, device):
    """Train ICL transformer on mixed-kappa data."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['wd'], betas=(0.9, 0.98))
    warmup, total = cfg.get('warmup', 1000), cfg['steps']
    lr_fn = lambda s: s / max(warmup, 1) if s < warmup else \
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    losses = []; t0 = time.time()
    for step in range(total):
        seq, y_q = generate_mixed_kappa_batch(
            cfg['batch_size'], cfg['n_support'], cfg['p'], KAPPAS, cfg['noise'], device)
        loss = ((model(seq) - y_q) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); losses.append(loss.item())
        if (step + 1) % cfg.get('log_every', 2000) == 0:
            avg = np.mean(losses[-cfg.get('log_every', 2000):])
            print(f"  step {step+1}/{total}  loss={avg:.5f}  "
                  f"elapsed={((time.time()-t0)/60):.1f}m")
    return losses

def train_layer_readouts(model, cfg, device, readout_steps=3000):
    """Train per-layer linear readout heads."""
    model.eval()
    readouts = nn.ModuleList([
        nn.Linear(model.d_model, 1).to(device) for _ in range(model.num_layers)])
    opt = torch.optim.Adam(readouts.parameters(), lr=1e-3)
    for step in range(readout_steps):
        seq, y_q = generate_mixed_kappa_batch(
            cfg['batch_size'], cfg['n_support'], cfg['p'], KAPPAS, cfg['noise'], device)
        with torch.no_grad():
            _, intermediates = model.forward_with_intermediates(seq)
        total_loss = sum(((readouts[l](intermediates[l+1]).squeeze(-1) - y_q)**2).mean()
                         for l in range(model.num_layers))
        opt.zero_grad(); total_loss.backward(); opt.step()
    return readouts

# -- Per-problem data collection ---------------------------------------------

def collect_per_problem_data(model, readouts, cfg, device, n_per_kappa=200):
    """Collect per-problem, per-layer predictions from model and all algorithms.

    Returns dict keyed by kappa with model_preds, algo_preds, y_true,
    algo_final_w (w_algo at final step), and problem_data (X_s, y_s, x_q).
    """
    model.eval()
    num_layers, p_dim = model.num_layers, cfg['p']
    lam = cfg.get('probe_lam', 0.1)
    all_data = {}
    for kappa in KAPPAS:
        print(f"  kappa={kappa:.0f}...")
        model_preds = [[] for _ in range(num_layers)]
        algo_preds = {n: [[] for _ in range(num_layers)] for n in ALGORITHMS}
        y_true_list, problem_data = [], []
        algo_final_w = {n: [] for n in ALGORITHMS}
        for start in range(0, n_per_kappa, 50):
            cur_bs = min(50, n_per_kappa - start)
            seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
                cur_bs, cfg['n_support'], p_dim, kappa, cfg['noise'], device)
            with torch.no_grad():
                _, intermediates = model.forward_with_intermediates(seq)
            x_s_np, y_s_np = x_s.cpu().numpy(), y_s.cpu().numpy()
            x_q_np, y_q_np = x_q.squeeze(1).cpu().numpy(), y_q.cpu().numpy()
            for i in range(cur_bs):
                X_i, y_i, xq_i, yq_i = x_s_np[i], y_s_np[i], x_q_np[i], float(y_q_np[i])
                y_true_list.append(yq_i)
                problem_data.append((X_i.copy(), y_i.copy(), xq_i.copy()))
                trajs = {}
                for name, fn in ALGORITHMS.items():
                    try: trajs[name] = fn(X_i, y_i, lam, num_layers)
                    except Exception: trajs[name] = [np.zeros(p_dim)] * (num_layers + 1)
                for name in ALGORITHMS:
                    algo_final_w[name].append(trajs[name][num_layers].copy())
                for l in range(num_layers):
                    with torch.no_grad():
                        model_preds[l].append(readouts[l](intermediates[l+1][i:i+1]).item())
                    for name in ALGORITHMS:
                        algo_preds[name][l].append(float(xq_i @ trajs[name][l+1]))
        all_data[kappa] = dict(model_preds=model_preds, algo_preds=algo_preds,
                               y_true=y_true_list, algo_final_w=algo_final_w,
                               problem_data=problem_data)
    return all_data

# -- Metric computation ------------------------------------------------------

def compute_rmse(all_data, num_layers):
    """Metric 1: RMSE(model_pred, algo_pred) per layer -- lower = better match."""
    results = {}
    for kappa, data in all_data.items():
        rmse = {n: [float(np.sqrt(np.mean((np.array(data['model_preds'][l])
                    - np.array(data['algo_preds'][n][l]))**2)))
                    for l in range(num_layers)] for n in ALGORITHMS}
        results[kappa] = {'per_layer': rmse,
                          'mean': {n: float(np.mean(rmse[n])) for n in ALGORITHMS}}
    return results

def compute_spearman_rank(all_data, num_layers):
    """Metric 2: Spearman correlation of per-problem squared errors."""
    results = {}
    for kappa, data in all_data.items():
        y_true = np.array(data['y_true']); sp = {}
        for name in ALGORITHMS:
            sp_layers = []
            for l in range(num_layers):
                m_err = (np.array(data['model_preds'][l]) - y_true) ** 2
                a_err = (np.array(data['algo_preds'][name][l]) - y_true) ** 2
                if np.std(m_err) < 1e-12 or np.std(a_err) < 1e-12:
                    sp_layers.append(0.0)
                else:
                    rho, _ = spearmanr(m_err, a_err)
                    sp_layers.append(float(rho) if np.isfinite(rho) else 0.0)
            sp[name] = sp_layers
        results[kappa] = {'per_layer': sp,
                          'mean': {n: float(np.mean(sp[n])) for n in ALGORITHMS}}
    return results

def compute_improvement_correlation(all_data, num_layers):
    """Metric 3: Pearson corr of error improvement at each layer transition."""
    results = {}
    for kappa, data in all_data.items():
        y_true = np.array(data['y_true']); impr = {}
        for name in ALGORITHMS:
            corrs = []
            for l in range(num_layers - 1):
                m_impr = ((np.array(data['model_preds'][l]) - y_true)**2
                        - (np.array(data['model_preds'][l+1]) - y_true)**2)
                a_impr = ((np.array(data['algo_preds'][name][l]) - y_true)**2
                        - (np.array(data['algo_preds'][name][l+1]) - y_true)**2)
                if np.std(m_impr) < 1e-12 or np.std(a_impr) < 1e-12:
                    corrs.append(0.0)
                else:
                    r = float(np.corrcoef(m_impr, a_impr)[0, 1])
                    corrs.append(r if np.isfinite(r) else 0.0)
            impr[name] = corrs
        results[kappa] = {'per_transition': impr,
                          'mean': {n: float(np.mean(impr[n])) for n in ALGORITHMS}}
    return results

def compute_pairwise_wins(all_data, num_layers):
    """Metric 4: fraction of problems where each algo is closest to model."""
    algo_names = list(ALGORITHMS.keys()); results = {}
    for kappa, data in all_data.items():
        n_prob = len(data['y_true'])
        wins = {n: np.zeros(num_layers) for n in ALGORITHMS}
        for l in range(num_layers):
            m = np.array(data['model_preds'][l])
            dists = np.stack([np.abs(m - np.array(data['algo_preds'][n][l]))
                              for n in algo_names])
            winners = np.argmin(dists, axis=0)
            for j, n in enumerate(algo_names):
                wins[n][l] = float(np.sum(winners == j))
        frac = {n: (wins[n] / n_prob).tolist() for n in ALGORITHMS}
        results[kappa] = {'per_layer': frac,
                          'mean': {n: float(np.mean(frac[n])) for n in ALGORITHMS}}
    return results

def compute_parameter_error(model, readouts, all_data, cfg, device,
                            n_probe_problems=20, n_probe_queries=50):
    """Metric 5: ||w_effective - w_algo|| at the final layer.

    Probes the model with varied query vectors (same support set) to extract
    the effective weight via least-squares, then compares with each algo's
    final-iterate weight.
    """
    num_layers, p_dim = model.num_layers, cfg['p']
    final_readout = readouts[num_layers - 1]
    results = {}
    for kappa, data in all_data.items():
        n_prob = len(data['y_true'])
        rng = np.random.default_rng(42)
        probe_idx = rng.choice(n_prob, size=min(n_probe_problems, n_prob), replace=False)
        sqrt_sig = np.sqrt(np.logspace(math.log10(max(kappa, 1.01)), 0, p_dim))
        param_err = {n: [] for n in ALGORITHMS}
        for idx in probe_idx:
            X_s, y_s, _ = data['problem_data'][idx]
            x_probes = rng.standard_normal((n_probe_queries, p_dim)) * sqrt_sig
            # Build sequences: shared support, varied query
            support_np = np.concatenate([X_s, y_s.reshape(-1, 1)], axis=1)
            supp_t = torch.tensor(support_np, dtype=torch.float32, device=device
                                  ).unsqueeze(0).expand(n_probe_queries, -1, -1)
            xq_t = torch.tensor(x_probes, dtype=torch.float32, device=device)
            query_t = torch.cat([xq_t, torch.zeros(n_probe_queries, 1, device=device)],
                                dim=1).unsqueeze(1)
            seq = torch.cat([supp_t, query_t], dim=1)
            with torch.no_grad():
                _, inters = model.forward_with_intermediates(seq)
                preds = final_readout(inters[num_layers]).squeeze(-1).cpu().numpy()
            # Least-squares: preds ~ x_probes @ w_eff + bias
            X_aug = np.column_stack([x_probes, np.ones(n_probe_queries)])
            w_eff = np.linalg.lstsq(X_aug, preds, rcond=None)[0][:p_dim]
            for name in ALGORITHMS:
                w_algo = data['algo_final_w'][name][idx]
                param_err[name].append(float(np.linalg.norm(w_eff - w_algo)))
        results[kappa] = {'per_problem': param_err,
                          'mean': {n: float(np.mean(param_err[n])) for n in ALGORITHMS}}
    return results

# -- Summary table for one metric -------------------------------------------

def summarize_metric(metric_name, results, higher_is_better=True):
    """Print a ranking table and check (a) GD separated, (b) top-3 clustered."""
    algo_names = list(ALGORITHMS.keys())
    kappas = sorted(k for k in results if isinstance(k, (int, float)))
    print(f"\n  --- {metric_name} ---")
    hdr = f"  {'kappa':>6} | " + " | ".join(f"{n:>11}" for n in algo_names) + " | Best"
    print(hdr); print("  " + "-" * len(hdr))
    for kappa in kappas:
        vals = results[kappa]['mean']
        best = (max if higher_is_better else min)(vals, key=vals.get)
        print(f"  {kappa:6.0f} | " +
              " | ".join(f"{vals[n]:>11.4f}" for n in algo_names) + f" | {best}")
    # Kappa-weighted aggregate
    norm = sum(math.log10(max(k, 1.1)) for k in kappas)
    agg = {n: sum(results[k]['mean'][n] * math.log10(max(k, 1.1)) for k in kappas) / norm
           for n in algo_names}
    ranked = sorted(agg, key=lambda n: -agg[n] if higher_is_better else agg[n])
    print(f"  Weighted ranking: {' > '.join(ranked[:3])} > ...")
    top3 = [agg[ranked[i]] for i in range(min(3, len(ranked)))]
    spread = max(top3) - min(top3)
    gd_sep = abs(agg['CG'] - agg['GD']) > 2 * spread if spread > 0 else True
    denom = abs(agg[ranked[0]]) if abs(agg[ranked[0]]) > 1e-12 else 1.0
    top3_cl = spread < 0.1 * denom
    print(f"  (a) GD clearly worse than CG class: {'YES' if gd_sep else 'no'}")
    print(f"  (b) Top 3 clustered: {'YES' if top3_cl else 'no'} (spread={spread:.4f})")
    return {'ranking': ranked, 'aggregated': agg,
            'gd_separated': gd_sep, 'top3_clustered': top3_cl}

# -- Combined summary figure -------------------------------------------------

def plot_metric_robustness(metric_results, out_dir):
    """2x3 figure: one panel per metric + ranking agreement text panel."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    algo_names = list(ALGORITHMS.keys())
    kappas = sorted(k for k in metric_results['rmse'] if isinstance(k, (int, float)))
    x = np.arange(len(kappas)); w = 0.12
    kappa_labels = [f'{k:.0f}' for k in kappas]

    # Line-plot panels
    line_panels = [
        (axes[0, 0], 'rmse', 'RMSE (model vs algo)', '1. RMSE on Predictions\n(lower = better match)'),
        (axes[0, 1], 'spearman', 'Spearman rho', '2. Spearman Rank Correlation\n(higher = error ranks agree)'),
        (axes[0, 2], 'improvement', 'Improvement correlation', '3. Per-Layer Improvement Corr\n(higher = same problems improve)'),
    ]
    for ax, key, ylabel, title in line_panels:
        for name in algo_names:
            vals = [metric_results[key][k]['mean'][name] for k in kappas]
            ax.plot(x, vals, '-o', color=ALGO_COLORS[name], label=name, ms=4)
        ax.set_xticks(x); ax.set_xticklabels(kappa_labels, fontsize=8)
        ax.set_xlabel('kappa'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # Bar-chart panels
    bar_panels = [
        (axes[1, 0], 'pairwise', 'Win fraction', '4. Pairwise Wins\n(higher = closest to model)'),
        (axes[1, 1], 'param_error', '||w_eff - w_algo||', '5. Final-Iterate Param Error\n(lower = weights agree)'),
    ]
    for ax, key, ylabel, title in bar_panels:
        for j, name in enumerate(algo_names):
            vals = [metric_results[key][k]['mean'][name] for k in kappas]
            ax.bar(x + (j - len(algo_names)/2 + 0.5) * w, vals, w,
                   color=ALGO_COLORS[name], label=name, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(kappa_labels, fontsize=8)
        ax.set_xlabel('kappa'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3, axis='y')

    # Text panel: ranking agreement
    ax = axes[1, 2]; ax.axis('off')
    summaries = metric_results.get('summaries', {})
    lines = ["RANKING AGREEMENT ACROSS METRICS\n"]
    for key, label in [('rmse','RMSE'), ('spearman','Spearman'),
                        ('improvement','Improv Corr'), ('pairwise','Pairwise'),
                        ('param_error','Param Err')]:
        if key in summaries:
            s = summaries[key]
            lines.append(f"{label:>12}: {' > '.join(s['ranking'][:3])}")
            lines.append(f"{'':>12}  GD sep={'Y' if s['gd_separated'] else 'N'}"
                         f"  Top3 clust={'Y' if s['top3_clustered'] else 'N'}")
    n_gd = sum(1 for s in summaries.values() if s.get('gd_separated'))
    n_cl = sum(1 for s in summaries.values() if s.get('top3_clustered'))
    lines += ["", f"GD clearly worse: {n_gd}/5 metrics",
              f"Top-3 clustered:  {n_cl}/5 metrics"]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=8,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Metric Robustness: Multi-Metric Agreement for Algorithm Identification',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{out_dir}/metric_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_dir}/metric_robustness.png")

# -- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Metric Robustness: multi-metric agreement for algorithm ID')
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--n-support', type=int, default=20)
    ap.add_argument('--p', type=int, default=10)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--num-layers', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--noise', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', type=str, default='docs/figures/metric_robustness')
    ap.add_argument('--n-per-kappa', type=int, default=200)
    ap.add_argument('--load', type=str, default=None,
                    help='Path to pretrained model checkpoint')
    ap.add_argument('--results-dir', type=str, default=None,
                    help='Path to previous algorithm_id.py output dir '
                         '(loads algo_id_results.json for comparison)')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = {'steps': args.steps, 'batch_size': args.batch_size,
           'n_support': args.n_support, 'p': args.p,
           'noise': args.noise, 'lr': args.lr, 'wd': 0.01,
           'warmup': min(1000, args.steps // 10),
           'log_every': max(1, args.steps // 25), 'probe_lam': 0.1}

    # Load baseline results from algorithm_id.py if available
    baseline = None
    if args.results_dir:
        jp = os.path.join(args.results_dir, 'algo_id_results.json')
        if os.path.exists(jp):
            with open(jp) as f: baseline = json.load(f)
            print(f"Loaded baseline from {jp}")
        else:
            print(f"Warning: {jp} not found, running without baseline")

    # Phase 1: Load or train model
    model = ICLTransformer(
        input_dim=args.p + 1, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, max_seq_len=args.n_support + 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.load and os.path.exists(args.load):
        print(f"\n{'='*60}\nLoading pretrained model from {args.load}\n{'='*60}")
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'cfg' in ckpt:
            for key in ['n_support', 'p', 'noise', 'probe_lam']:
                if key in ckpt['cfg']: cfg[key] = ckpt['cfg'][key]
        print(f"  Parameters: {n_params:,}")
    else:
        print(f"\n{'='*60}\nPhase 1: Training ({args.num_layers}L, d={args.d_model}, "
              f"{args.steps} steps)\n{'='*60}")
        t0 = time.time(); losses = train_model(model, cfg, device)
        print(f"  Training: {(time.time()-t0)/60:.1f} min, "
              f"final loss: {np.mean(losses[-1000:]):.5f}")

    # Phase 2: Train per-layer readouts
    print(f"\n{'='*60}\nPhase 2: Training per-layer readout heads\n{'='*60}")
    t0 = time.time()
    readouts = train_layer_readouts(model, cfg, device, readout_steps=3000)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Phase 3: Collect per-problem data
    print(f"\n{'='*60}\nPhase 3: Collecting per-problem data "
          f"({args.n_per_kappa}/kappa, {len(KAPPAS)} kappas)\n{'='*60}")
    t0 = time.time()
    all_data = collect_per_problem_data(
        model, readouts, cfg, device, n_per_kappa=args.n_per_kappa)
    print(f"  Data collection: {time.time()-t0:.1f}s")

    # Phase 4: Compute all metrics
    num_layers = args.num_layers
    print(f"\n{'='*60}\nPhase 4: Computing robustness metrics\n{'='*60}")
    metric_fns = [
        ('rmse',        lambda: compute_rmse(all_data, num_layers)),
        ('spearman',    lambda: compute_spearman_rank(all_data, num_layers)),
        ('improvement', lambda: compute_improvement_correlation(all_data, num_layers)),
        ('pairwise',    lambda: compute_pairwise_wins(all_data, num_layers)),
        ('param_error', lambda: compute_parameter_error(
            model, readouts, all_data, cfg, device)),
    ]
    metric_bundle = {}
    for i, (name, fn) in enumerate(metric_fns, 1):
        print(f"  {i}/5  {name}...")
        metric_bundle[name] = fn()

    # Phase 5: Summary tables
    print(f"\n{'='*60}\nRESULTS: Metric Robustness Analysis\n{'='*60}")
    summaries = {}
    metric_cfg = [
        ('rmse', 'RMSE on Predictions (lower = better)', False),
        ('spearman', 'Spearman Rank Correlation (higher = better)', True),
        ('improvement', 'Per-Layer Improvement Correlation (higher = better)', True),
        ('pairwise', 'Pairwise Wins (higher = better)', True),
        ('param_error', 'Final-Iterate Parameter Error (lower = better)', False),
    ]
    for key, label, higher in metric_cfg:
        summaries[key] = summarize_metric(label, metric_bundle[key], higher)

    # Cross-reference with baseline
    if baseline and 'weighted_r2' in baseline:
        print(f"\n  --- Baseline comparison (from algorithm_id.py) ---")
        for name, r2 in sorted(baseline['weighted_r2'].items(), key=lambda x: -x[1]):
            ci = baseline.get('weighted_ci', {}).get(name, 0.0)
            print(f"    {name:>15}: {r2:.4f} +/- {ci:.4f}")

    # Cross-metric agreement
    print(f"\n{'='*60}\nCROSS-METRIC AGREEMENT\n{'='*60}")
    n_gd_sep = sum(1 for s in summaries.values() if s['gd_separated'])
    n_top3_cl = sum(1 for s in summaries.values() if s['top3_clustered'])
    top1_counts = {}
    for s in summaries.values():
        top1_counts[s['ranking'][0]] = top1_counts.get(s['ranking'][0], 0) + 1
    top1 = max(top1_counts, key=top1_counts.get)

    print(f"\n  GD clearly worse than CG class: {n_gd_sep}/5 metrics")
    print(f"  Top-3 (CG/PCG/PGD) clustered:   {n_top3_cl}/5 metrics")
    print(f"  Most frequent #1 algorithm:      {top1} ({top1_counts[top1]}/5 metrics)")
    print(f"\n  CONCLUSION: ", end="")
    if n_gd_sep >= 4:
        print(f"GD is robustly separated from the CG class ({n_gd_sep}/5).")
    else:
        print(f"GD separation is metric-dependent ({n_gd_sep}/5).")
    if n_top3_cl >= 3:
        print(f"  The top CG-class algorithms are clustered across metrics,")
        print(f"  consistent with the model implementing a CG-class algorithm")
        print(f"  whose exact variant is hard to distinguish at this scale.")
    else:
        print(f"  Some metrics can differentiate within the CG class;")
        print(f"  see individual rankings for details.")

    # Phase 6: Plot
    print(f"\n  Generating combined summary figure...")
    metric_bundle['summaries'] = summaries
    plot_metric_robustness(metric_bundle, args.out_dir)

    # Save JSON results
    save_data = {
        'config': cfg, 'algorithms': list(ALGORITHMS.keys()),
        'metrics': {}, 'summaries': {},
        'cross_metric': {'gd_separated_count': n_gd_sep,
                         'top3_clustered_count': n_top3_cl,
                         'most_common_top1': top1},
    }
    for mname in ['rmse', 'spearman', 'improvement', 'pairwise', 'param_error']:
        save_data['metrics'][mname] = {
            f'kappa_{k:.0f}': {'mean': metric_bundle[mname][k]['mean']}
            for k in sorted(k for k in metric_bundle[mname] if isinstance(k, (int, float)))}
    for key, s in summaries.items():
        save_data['summaries'][key] = {
            'ranking': s['ranking'], 'aggregated': s['aggregated'],
            'gd_separated': s['gd_separated'], 'top3_clustered': s['top3_clustered']}
    json_out = f'{args.out_dir}/metric_robustness_results.json'
    with open(json_out, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to {json_out}")


if __name__ == '__main__':
    main()
