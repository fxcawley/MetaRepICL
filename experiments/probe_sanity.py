"""Probe Sanity-Check Suite: Validate or demote probe findings.

The probe-behavior mismatch (GD probes > CG probes despite CG-like convergence)
is the most vulnerable claim in the paper. Before publishing it, we need:

1. Capacity sweep: linear vs MLP probes (can nonlinear probes find CG states?)
2. Layer-wise bootstrap CIs on probe cosine similarity
3. Control targets: ridge solution w*, raw y_support (positive sanity checks)
4. Basis control: random rotation of targets (pipeline sanity check)
5. Distribution shift: train probes on kappa=10, test on kappa=100

Usage:
    python experiments/probe_sanity.py --load docs/figures/trained_mixed/model_mixed.pt
    python experiments/probe_sanity.py --steps 5000 --n-probe-problems 100  # quick
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
    KAPPAS, generate_batch_with_cov, generate_mixed_kappa_batch,
    train_model, train_layer_readouts,
)


# ---------------------------------------------------------------------------
# CG / GD reference trajectories (feature space)
# ---------------------------------------------------------------------------

def cg_trajectory_feature(X, y, lam, steps):
    """CG on (X^T X + lam I) w = X^T y. Returns list of w vectors."""
    p = X.shape[1]
    B = X.T @ X + lam * np.eye(p)
    rhs = X.T @ y
    w = np.zeros(p)
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


def gd_trajectory_feature(X, y, lam, steps):
    """GD on (X^T X + lam I) w = X^T y. Returns list of w vectors."""
    p = X.shape[1]
    B = X.T @ X + lam * np.eye(p)
    rhs = X.T @ y
    eigs = np.linalg.eigvalsh(B)
    eta = 2.0 / (eigs[-1] + eigs[0])
    w = np.zeros(p)
    traj = [w.copy()]
    for _ in range(steps):
        grad = B @ w - rhs
        w = w - eta * grad
        traj.append(w.copy())
    return traj


# ---------------------------------------------------------------------------
# MLP Probe
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """MLP probe for nonlinear state recovery."""
    def __init__(self, input_dim, output_dim, hidden_dims=(64,)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_linear_probe(X_train, y_train, X_test, y_test, lam_reg=1e-3):
    """Ridge regression probe. Returns cosine similarity on test set."""
    XtX = X_train.T @ X_train
    reg = lam_reg * np.eye(XtX.shape[0])
    try:
        W = np.linalg.solve(XtX + reg, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        return 0.0
    pred = X_test @ W
    p_flat = pred.flatten()
    y_flat = y_test.flatten()
    norms = np.linalg.norm(p_flat) * np.linalg.norm(y_flat)
    if norms < 1e-12:
        return 0.0
    return float(np.dot(p_flat, y_flat) / norms)


def fit_mlp_probe(X_train, y_train, X_test, y_test, hidden_dims=(64,),
                  epochs=200, lr=1e-3):
    """Train MLP probe. Returns cosine similarity on test set."""
    device = 'cpu'
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.float32, device=device)

    model = MLPProbe(X_tr.shape[1], y_tr.shape[1], hidden_dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        pred = model(X_tr)
        loss = ((pred - y_tr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_te).cpu().numpy()
    p_flat = pred.flatten()
    y_flat = y_test.flatten()
    norms = np.linalg.norm(p_flat) * np.linalg.norm(y_flat)
    if norms < 1e-12:
        return 0.0
    return float(np.dot(p_flat, y_flat) / norms)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_probe_data(model, cfg, device, n_problems=200, kappa=None):
    """Collect (activations, CG states, GD states, w_star, y_support) for probing."""
    model.eval()
    num_layers = model.num_layers
    n_support = cfg['n_support']
    p_dim = cfg['p']
    lam = cfg.get('probe_lam', 0.1)

    all_acts = {l: [] for l in range(num_layers + 1)}
    all_cg = {l: [] for l in range(1, num_layers + 1)}
    all_gd = {l: [] for l in range(1, num_layers + 1)}
    all_wstar = []
    all_ysup = []

    bs = 50
    for start in range(0, n_problems, bs):
        cur_bs = min(bs, n_problems - start)
        if kappa is not None:
            seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
                cur_bs, n_support, p_dim, kappa, cfg['noise'], device)
        else:
            seq, y_q = generate_mixed_kappa_batch(
                cur_bs, n_support, p_dim, KAPPAS, cfg['noise'], device)
            x_s = seq[:, :n_support, :p_dim]
            y_s = seq[:, :n_support, p_dim]

        with torch.no_grad():
            _, intermediates = model.forward_with_intermediates(seq)

        for l in range(num_layers + 1):
            all_acts[l].append(intermediates[l].cpu().numpy())

        x_s_np = x_s.cpu().numpy()
        y_s_np = y_s.cpu().numpy()

        for i in range(cur_bs):
            X_i = x_s_np[i]
            y_vec = y_s_np[i]
            all_ysup.append(y_vec)

            # Ridge solution
            B = X_i.T @ X_i + lam * np.eye(p_dim)
            rhs = X_i.T @ y_vec
            try:
                w_star = np.linalg.solve(B, rhs)
            except np.linalg.LinAlgError:
                w_star = np.zeros(p_dim)
            all_wstar.append(w_star)

            cg_traj = cg_trajectory_feature(X_i, y_vec, lam, num_layers)
            gd_traj = gd_trajectory_feature(X_i, y_vec, lam, num_layers)
            for l in range(1, num_layers + 1):
                all_cg[l].append(cg_traj[l])
                all_gd[l].append(gd_traj[l])

    # Stack
    for l in range(num_layers + 1):
        all_acts[l] = np.concatenate(all_acts[l], axis=0)
    for l in range(1, num_layers + 1):
        all_cg[l] = np.array(all_cg[l])
        all_gd[l] = np.array(all_gd[l])

    return {
        'acts': all_acts,
        'cg': all_cg,
        'gd': all_gd,
        'wstar': np.array(all_wstar),
        'ysup': np.array(all_ysup),
    }


# ---------------------------------------------------------------------------
# Experiment 1: Capacity sweep
# ---------------------------------------------------------------------------

def run_capacity_sweep(data, num_layers):
    """Compare linear vs MLP probes of different sizes."""
    n = data['acts'][0].shape[0]
    split = int(0.8 * n)

    configs = [
        ('Linear', None),
        ('MLP-64', (64,)),
        ('MLP-256', (256,)),
        ('MLP-128-64', (128, 64)),
    ]

    results = {}
    for probe_name, hidden in configs:
        print(f"    {probe_name}...")
        cg_sims = []
        gd_sims = []
        ctrl_sims = []

        for l in range(1, num_layers + 1):
            X = data['acts'][l]
            X_tr, X_te = X[:split], X[split:]

            cg_tr, cg_te = data['cg'][l][:split], data['cg'][l][split:]
            gd_tr, gd_te = data['gd'][l][:split], data['gd'][l][split:]
            rng = np.random.default_rng(42 + l)
            ctrl = rng.standard_normal(data['cg'][l].shape)
            ctrl_tr, ctrl_te = ctrl[:split], ctrl[split:]

            if hidden is None:
                cg_sims.append(fit_linear_probe(X_tr, cg_tr, X_te, cg_te))
                gd_sims.append(fit_linear_probe(X_tr, gd_tr, X_te, gd_te))
                ctrl_sims.append(fit_linear_probe(X_tr, ctrl_tr, X_te, ctrl_te))
            else:
                cg_sims.append(fit_mlp_probe(X_tr, cg_tr, X_te, cg_te, hidden))
                gd_sims.append(fit_mlp_probe(X_tr, gd_tr, X_te, gd_te, hidden))
                ctrl_sims.append(fit_mlp_probe(X_tr, ctrl_tr, X_te, ctrl_te, hidden))

        results[probe_name] = {
            'cg': cg_sims, 'gd': gd_sims, 'ctrl': ctrl_sims,
            'cg_mean': float(np.mean(cg_sims)),
            'gd_mean': float(np.mean(gd_sims)),
            'ctrl_mean': float(np.mean(ctrl_sims)),
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Bootstrap CIs
# ---------------------------------------------------------------------------

def run_bootstrap_cis(data, num_layers, n_bootstrap=200):
    """Bootstrap CIs on per-layer probe cosine similarity."""
    n = data['acts'][0].shape[0]
    split = int(0.8 * n)

    cg_cis = []
    gd_cis = []
    for l in range(1, num_layers + 1):
        X = data['acts'][l]
        X_tr, X_te = X[:split], X[split:]
        cg_tr, cg_te = data['cg'][l][:split], data['cg'][l][split:]
        gd_tr, gd_te = data['gd'][l][:split], data['gd'][l][split:]

        boot_rng = np.random.default_rng(42 + l)
        n_test = X_te.shape[0]

        cg_boots = []
        gd_boots = []
        for _ in range(n_bootstrap):
            idx = boot_rng.choice(n_test, size=n_test, replace=True)
            cg_boots.append(fit_linear_probe(X_tr, cg_tr, X_te[idx], cg_te[idx]))
            gd_boots.append(fit_linear_probe(X_tr, gd_tr, X_te[idx], gd_te[idx]))

        cg_cis.append({
            'mean': float(np.mean(cg_boots)),
            'ci': float(1.96 * np.std(cg_boots)),
        })
        gd_cis.append({
            'mean': float(np.mean(gd_boots)),
            'ci': float(1.96 * np.std(gd_boots)),
        })

    return {'cg': cg_cis, 'gd': gd_cis}


# ---------------------------------------------------------------------------
# Experiment 3: Control targets
# ---------------------------------------------------------------------------

def run_control_targets(data, num_layers):
    """Probe for control targets that should be recoverable."""
    n = data['acts'][0].shape[0]
    split = int(0.8 * n)

    # w* should be recoverable at late layers (model is trying to solve for w*)
    wstar_sims = []
    ysup_sims = []
    for l in range(1, num_layers + 1):
        X = data['acts'][l]
        X_tr, X_te = X[:split], X[split:]

        # w* probe
        ws_tr, ws_te = data['wstar'][:split], data['wstar'][split:]
        wstar_sims.append(fit_linear_probe(X_tr, ws_tr, X_te, ws_te))

        # y_support probe (should be trivially available)
        ys_tr, ys_te = data['ysup'][:split], data['ysup'][split:]
        ysup_sims.append(fit_linear_probe(X_tr, ys_tr, X_te, ys_te))

    return {
        'wstar': wstar_sims,
        'ysup': ysup_sims,
        'wstar_mean': float(np.mean(wstar_sims)),
        'ysup_mean': float(np.mean(ysup_sims)),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Basis control
# ---------------------------------------------------------------------------

def run_basis_control(data, num_layers, seed=123):
    """Rotate CG targets by random orthogonal Q. Cosine sim should be same."""
    n = data['acts'][0].shape[0]
    split = int(0.8 * n)
    rng = np.random.default_rng(seed)

    original_sims = []
    rotated_sims = []
    for l in range(1, num_layers + 1):
        X = data['acts'][l]
        X_tr, X_te = X[:split], X[split:]
        cg_tr, cg_te = data['cg'][l][:split], data['cg'][l][split:]

        # Original
        original_sims.append(fit_linear_probe(X_tr, cg_tr, X_te, cg_te))

        # Random orthogonal rotation
        p_dim = cg_tr.shape[1]
        Q, _ = np.linalg.qr(rng.standard_normal((p_dim, p_dim)))
        cg_tr_rot = cg_tr @ Q.T
        cg_te_rot = cg_te @ Q.T
        rotated_sims.append(fit_linear_probe(X_tr, cg_tr_rot, X_te, cg_te_rot))

    return {
        'original': original_sims,
        'rotated': rotated_sims,
        'original_mean': float(np.mean(original_sims)),
        'rotated_mean': float(np.mean(rotated_sims)),
        'consistent': abs(np.mean(original_sims) - np.mean(rotated_sims)) < 0.05,
    }


# ---------------------------------------------------------------------------
# Experiment 5: Distribution shift
# ---------------------------------------------------------------------------

def run_distribution_shift(model, cfg, device, num_layers, n_problems=200):
    """Train probes on kappa=10, test on kappa=100."""
    print("    Collecting kappa=10 data (train)...")
    train_data = collect_probe_data(model, cfg, device, n_problems, kappa=10.0)
    print("    Collecting kappa=100 data (test)...")
    test_data = collect_probe_data(model, cfg, device, n_problems, kappa=100.0)

    n_tr = train_data['acts'][0].shape[0]
    n_te = test_data['acts'][0].shape[0]

    in_dist_sims = []
    ood_sims = []
    for l in range(1, num_layers + 1):
        # In-distribution: train and test both from kappa=10
        split = int(0.8 * n_tr)
        X_tr = train_data['acts'][l][:split]
        X_te_in = train_data['acts'][l][split:]
        cg_tr = train_data['cg'][l][:split]
        cg_te_in = train_data['cg'][l][split:]
        in_dist_sims.append(fit_linear_probe(X_tr, cg_tr, X_te_in, cg_te_in))

        # OOD: train on kappa=10, test on kappa=100
        X_te_ood = test_data['acts'][l]
        cg_te_ood = test_data['cg'][l]
        ood_sims.append(fit_linear_probe(X_tr, cg_tr, X_te_ood, cg_te_ood))

    return {
        'in_dist': in_dist_sims,
        'ood': ood_sims,
        'in_dist_mean': float(np.mean(in_dist_sims)),
        'ood_mean': float(np.mean(ood_sims)),
        'generalize': np.mean(ood_sims) > 0.5 * np.mean(in_dist_sims),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all_results(capacity, bootstrap, controls, basis, shift, num_layers, out_dir):
    """Comprehensive probe sanity check figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    layers = list(range(1, num_layers + 1))

    # 1. Capacity sweep
    ax = axes[0, 0]
    for probe_name, res in capacity.items():
        ax.plot(layers, res['cg'], '-o', markersize=3,
                label=f'{probe_name} CG (mean={res["cg_mean"]:.3f})')
        ax.plot(layers, res['gd'], '--s', markersize=3,
                label=f'{probe_name} GD (mean={res["gd_mean"]:.3f})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Probe Capacity Sweep')
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. Bootstrap CIs
    ax = axes[0, 1]
    cg_means = [c['mean'] for c in bootstrap['cg']]
    cg_cis = [c['ci'] for c in bootstrap['cg']]
    gd_means = [c['mean'] for c in bootstrap['gd']]
    gd_cis = [c['ci'] for c in bootstrap['gd']]
    ax.errorbar(layers, cg_means, yerr=cg_cis, marker='o', label='CG', color='green', capsize=3)
    ax.errorbar(layers, gd_means, yerr=gd_cis, marker='s', label='GD', color='red', capsize=3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Layer-wise Probe CIs (95% bootstrap)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Control targets
    ax = axes[0, 2]
    ax.plot(layers, controls['wstar'], '-o', color='blue',
            label=f'w* (mean={controls["wstar_mean"]:.3f})')
    ax.plot(layers, controls['ysup'], '--s', color='orange',
            label=f'y_support (mean={controls["ysup_mean"]:.3f})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Control Targets (sanity check)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Basis control
    ax = axes[1, 0]
    ax.plot(layers, basis['original'], '-o', label='Original CG', color='green')
    ax.plot(layers, basis['rotated'], '--s', label='Rotated CG (Q@z)', color='purple')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    consistent = "YES" if basis['consistent'] else "NO"
    ax.set_title(f'Basis Control (consistent: {consistent})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Distribution shift
    ax = axes[1, 1]
    ax.plot(layers, shift['in_dist'], '-o', label=f'In-dist kappa=10 ({shift["in_dist_mean"]:.3f})',
            color='green')
    ax.plot(layers, shift['ood'], '--s', label=f'OOD kappa=100 ({shift["ood_mean"]:.3f})',
            color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    generalize = "YES" if shift['generalize'] else "NO"
    ax.set_title(f'Distribution Shift (generalizes: {generalize})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary verdict
    ax = axes[1, 2]
    ax.axis('off')

    # Determine verdict
    linear_cg = capacity.get('Linear', {}).get('cg_mean', 0)
    mlp256_cg = capacity.get('MLP-256', {}).get('cg_mean', 0)
    mlp_helps = mlp256_cg > linear_cg + 0.05

    checks = {
        'MLP recovers CG better': mlp_helps,
        'Bootstrap CIs non-overlapping': any(
            bootstrap['gd'][l]['mean'] - bootstrap['gd'][l]['ci'] >
            bootstrap['cg'][l]['mean'] + bootstrap['cg'][l]['ci']
            for l in range(num_layers)),
        'w* recoverable at late layers': controls['wstar_mean'] > 0.3,
        'y_support recoverable': controls['ysup_mean'] > 0.3,
        'Basis rotation consistent': basis['consistent'],
        'Probes generalize across kappa': shift['generalize'],
    }

    text = "PROBE SANITY VERDICT\n" + "=" * 30 + "\n\n"
    passed = 0
    for check, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        text += f"  [{status:4}] {check}\n"
        if ok:
            passed += 1
    text += f"\n  {passed}/{len(checks)} checks passed\n\n"
    if passed >= 4:
        text += "  VERDICT: Probes ADEQUATE\n  (can report findings)"
    else:
        text += "  VERDICT: Probes UNRELIABLE\n  (demote to exploratory)"

    ax.text(0.05, 0.5, text, fontsize=9, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Probe Sanity-Check Suite', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/probe_sanity.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Probe Sanity-Check Suite')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-support', type=int, default=20)
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='docs/figures/probe_sanity')
    parser.add_argument('--n-probe-problems', type=int, default=300)
    parser.add_argument('--load', type=str, default=None)
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

    # ---- Load or train model ----
    model = ICLTransformer(
        input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, max_seq_len=args.n_support + 1,
    ).to(device)

    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}")
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'cfg' in ckpt:
            for key in ['n_support', 'p', 'noise', 'probe_lam']:
                if key in ckpt['cfg']:
                    cfg[key] = ckpt['cfg'][key]
    else:
        print(f"Training model ({args.steps} steps)...")
        train_model(model, cfg, device)

    num_layers = model.num_layers

    # ---- Collect probe data (mixed kappa) ----
    print(f"\nCollecting probe data ({args.n_probe_problems} problems)...")
    t0 = time.time()
    data = collect_probe_data(model, cfg, device, args.n_probe_problems)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Experiment 1: Capacity sweep ----
    print(f"\n{'='*60}")
    print("Experiment 1: Probe Capacity Sweep")
    print(f"{'='*60}")
    t0 = time.time()
    capacity = run_capacity_sweep(data, num_layers)
    print(f"  Done in {time.time()-t0:.1f}s")
    for pname, res in capacity.items():
        print(f"  {pname:>12}: CG={res['cg_mean']:.3f}, GD={res['gd_mean']:.3f}, "
              f"Ctrl={res['ctrl_mean']:.3f}")

    # ---- Experiment 2: Bootstrap CIs ----
    print(f"\n{'='*60}")
    print("Experiment 2: Layer-wise Bootstrap CIs")
    print(f"{'='*60}")
    t0 = time.time()
    bootstrap = run_bootstrap_cis(data, num_layers)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Experiment 3: Control targets ----
    print(f"\n{'='*60}")
    print("Experiment 3: Control Targets")
    print(f"{'='*60}")
    t0 = time.time()
    controls = run_control_targets(data, num_layers)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  w* probe mean: {controls['wstar_mean']:.3f}")
    print(f"  y_support probe mean: {controls['ysup_mean']:.3f}")

    # ---- Experiment 4: Basis control ----
    print(f"\n{'='*60}")
    print("Experiment 4: Basis Control (rotation invariance)")
    print(f"{'='*60}")
    t0 = time.time()
    basis = run_basis_control(data, num_layers)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Original CG mean: {basis['original_mean']:.3f}")
    print(f"  Rotated CG mean:  {basis['rotated_mean']:.3f}")
    print(f"  Consistent: {'YES' if basis['consistent'] else 'NO'}")

    # ---- Experiment 5: Distribution shift ----
    print(f"\n{'='*60}")
    print("Experiment 5: Distribution Shift (kappa=10 -> kappa=100)")
    print(f"{'='*60}")
    t0 = time.time()
    shift = run_distribution_shift(model, cfg, device, num_layers,
                                   n_problems=args.n_probe_problems)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  In-distribution mean:  {shift['in_dist_mean']:.3f}")
    print(f"  Out-of-distribution mean: {shift['ood_mean']:.3f}")
    print(f"  Generalizes: {'YES' if shift['generalize'] else 'NO'}")

    # ---- Plot ----
    print(f"\nGenerating plots...")
    plot_all_results(capacity, bootstrap, controls, basis, shift, num_layers, args.out_dir)

    # ---- Save ----
    save_data = {
        'capacity_sweep': {k: {'cg_mean': v['cg_mean'], 'gd_mean': v['gd_mean'],
                               'ctrl_mean': v['ctrl_mean']}
                          for k, v in capacity.items()},
        'controls': {'wstar_mean': controls['wstar_mean'],
                    'ysup_mean': controls['ysup_mean']},
        'basis': {'original_mean': basis['original_mean'],
                 'rotated_mean': basis['rotated_mean'],
                 'consistent': basis['consistent']},
        'distribution_shift': {'in_dist_mean': shift['in_dist_mean'],
                              'ood_mean': shift['ood_mean'],
                              'generalize': shift['generalize']},
    }
    with open(f'{args.out_dir}/probe_sanity_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
