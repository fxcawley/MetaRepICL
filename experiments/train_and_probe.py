"""Train a Transformer on ICL regression and probe for CG-like structure.

This is the 'critical experiment' identified in the external review (W1):
training an actual neural network via SGD and probing its internal states
for CG variables, comparing convergence rates against CG and GD theory.

Usage:
    python experiments/train_and_probe.py                  # full run
    python experiments/train_and_probe.py --steps 5000     # quick test
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

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_batch(batch_size, n_support, p, noise=0.1, device='cpu'):
    """Generate a batch of random linear regression ICL tasks.

    Each task: w ~ N(0, I/p), x ~ N(0, I), y = x^T w + noise.
    Returns (seq, y_q) where seq is (B, n_support+1, p+1) and y_q is (B,).
    """
    w = torch.randn(batch_size, p, 1, device=device) / math.sqrt(p)
    x_s = torch.randn(batch_size, n_support, p, device=device)
    y_s = (x_s @ w).squeeze(-1) + noise * torch.randn(batch_size, n_support, device=device)
    x_q = torch.randn(batch_size, 1, p, device=device)
    y_q = (x_q @ w).squeeze(-1).squeeze(-1)

    support_tokens = torch.cat([x_s, y_s.unsqueeze(-1)], dim=-1)
    query_token = torch.cat([x_q, torch.zeros(batch_size, 1, 1, device=device)], dim=-1)
    seq = torch.cat([support_tokens, query_token], dim=1)
    return seq, y_q


def generate_batch_with_cov(batch_size, n_support, p, cond, noise=0.1, device='cpu'):
    """Generate batch with controlled feature covariance condition number."""
    sigmas = torch.logspace(math.log10(cond), 0, p, device=device)
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

# ---------------------------------------------------------------------------
# CG / GD reference trajectories (numpy, per-problem)
# ---------------------------------------------------------------------------

def cg_trajectory(K, y, lam, steps):
    """Run CG on (K + lam I) alpha = y. Returns list of alpha vectors."""
    n = K.shape[0]
    A = K + lam * np.eye(n)
    alpha = np.zeros(n)
    r = y.copy()
    p = r.copy()
    traj = [alpha.copy()]
    for _ in range(steps):
        Ap = A @ p
        rr = r @ r
        gamma = rr / (p @ Ap + 1e-30)
        alpha = alpha + gamma * p
        r = r - gamma * Ap
        beta = (r @ r) / (rr + 1e-30)
        p = r + beta * p
        traj.append(alpha.copy())
    return traj


def gd_trajectory(K, y, lam, steps):
    """Run GD on (K + lam I) alpha = y with optimal fixed step size."""
    n = K.shape[0]
    A = K + lam * np.eye(n)
    eigs = np.linalg.eigvalsh(A)
    eta = 2.0 / (eigs[-1] + eigs[0])
    alpha = np.zeros(n)
    traj = [alpha.copy()]
    for _ in range(steps):
        grad = A @ alpha - y
        alpha = alpha - eta * grad
        traj.append(alpha.copy())
    return traj

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, cfg, device):
    """Train the ICL transformer. Returns training loss history."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['wd'], betas=(0.9, 0.98))

    # Cosine schedule with warmup
    warmup = cfg.get('warmup', 1000)
    total = cfg['steps']
    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    losses = []
    t0 = time.time()
    for step in range(total):
        seq, y_q = generate_batch(cfg['batch_size'], cfg['n_support'],
                                  cfg['p'], cfg['noise'], device)
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

# ---------------------------------------------------------------------------
# Train per-layer readout heads (for convergence analysis)
# ---------------------------------------------------------------------------

def train_layer_readouts(model, cfg, device, readout_steps=2000):
    """Train independent linear readout heads for each layer."""
    model.eval()
    num_layers = model.num_layers
    readouts = nn.ModuleList([
        nn.Linear(model.d_model, 1).to(device) for _ in range(num_layers)
    ])
    opt = torch.optim.Adam(readouts.parameters(), lr=1e-3)

    for step in range(readout_steps):
        seq, y_q = generate_batch(cfg['batch_size'], cfg['n_support'],
                                  cfg['p'], cfg['noise'], device)
        with torch.no_grad():
            _, intermediates = model.forward_with_intermediates(seq)

        total_loss = 0.0
        for l in range(num_layers):
            h = intermediates[l + 1]  # after layer l (1-indexed)
            pred = readouts[l](h).squeeze(-1)
            total_loss = total_loss + ((pred - y_q) ** 2).mean()

        opt.zero_grad()
        total_loss.backward()
        opt.step()

    model._layer_readouts = readouts
    return readouts

# ---------------------------------------------------------------------------
# Probing for CG / GD states
# ---------------------------------------------------------------------------

def collect_probe_data(model, cfg, device, n_problems=500):
    """Collect (activations, CG states, GD states) for probing."""
    model.eval()
    num_layers = model.num_layers
    n_support = cfg['n_support']
    p_dim = cfg['p']
    lam = cfg.get('probe_lam', 0.1)

    # Storage: per-layer activations and reference states
    all_acts = {l: [] for l in range(num_layers + 1)}
    all_cg = {l: [] for l in range(1, num_layers + 1)}
    all_gd = {l: [] for l in range(1, num_layers + 1)}

    batch = 50
    for start in range(0, n_problems, batch):
        bs = min(batch, n_problems - start)
        seq, y_q = generate_batch(bs, n_support, p_dim, cfg['noise'], device)

        with torch.no_grad():
            _, intermediates = model.forward_with_intermediates(seq)

        # Store activations
        for l in range(num_layers + 1):
            all_acts[l].append(intermediates[l].cpu().numpy())

        # Compute CG and GD reference trajectories per problem
        x_s_np = seq[:, :n_support, :p_dim].cpu().numpy()
        y_s_np = seq[:, :n_support, p_dim].cpu().numpy()

        for i in range(bs):
            K = x_s_np[i] @ x_s_np[i].T
            y = y_s_np[i]
            cg_traj = cg_trajectory(K, y, lam, num_layers)
            gd_traj = gd_trajectory(K, y, lam, num_layers)
            for l in range(1, num_layers + 1):
                all_cg[l].append(cg_traj[l])
                all_gd[l].append(gd_traj[l])

    # Stack
    for l in range(num_layers + 1):
        all_acts[l] = np.concatenate(all_acts[l], axis=0)
    for l in range(1, num_layers + 1):
        all_cg[l] = np.array(all_cg[l])
        all_gd[l] = np.array(all_gd[l])

    return all_acts, all_cg, all_gd


def fit_probe(X_train, y_train, X_test, y_test, lam_reg=1e-3):
    """Fit a ridge regression probe, return cosine similarity on test set."""
    XtX = X_train.T @ X_train
    reg = lam_reg * np.eye(XtX.shape[0])
    try:
        W = np.linalg.solve(XtX + reg, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        return 0.0
    pred = X_test @ W
    # Cosine similarity (flattened)
    p_flat = pred.flatten()
    y_flat = y_test.flatten()
    norms = np.linalg.norm(p_flat) * np.linalg.norm(y_flat)
    if norms < 1e-12:
        return 0.0
    return float(np.dot(p_flat, y_flat) / norms)


def run_probe_analysis(all_acts, all_cg, all_gd, num_layers):
    """Fit linear probes per layer, comparing CG vs GD vs random targets."""
    n = all_acts[0].shape[0]
    split = int(0.8 * n)

    cg_sims = []
    gd_sims = []
    ctrl_sims = []

    for l in range(1, num_layers + 1):
        X = all_acts[l]
        X_train, X_test = X[:split], X[split:]
        cg_targets = all_cg[l]
        gd_targets = all_gd[l]

        cg_train, cg_test = cg_targets[:split], cg_targets[split:]
        gd_train, gd_test = gd_targets[:split], gd_targets[split:]
        rng = np.random.default_rng(42 + l)
        ctrl = rng.standard_normal(cg_targets.shape)
        ctrl_train, ctrl_test = ctrl[:split], ctrl[split:]

        cg_sims.append(fit_probe(X_train, cg_train, X_test, cg_test))
        gd_sims.append(fit_probe(X_train, gd_train, X_test, gd_test))
        ctrl_sims.append(fit_probe(X_train, ctrl_train, X_test, ctrl_test))

    return cg_sims, gd_sims, ctrl_sims

# ---------------------------------------------------------------------------
# Convergence rate analysis
# ---------------------------------------------------------------------------

def run_convergence_analysis(model, readouts, cfg, device, kappas=None):
    """Measure per-layer prediction error for different condition numbers."""
    if kappas is None:
        kappas = [1.0, 5.0, 10.0, 50.0]
    model.eval()
    num_layers = model.num_layers
    n_support = cfg['n_support']
    p_dim = cfg['p']
    lam = cfg.get('probe_lam', 0.1)
    n_test = 200

    results = {}
    for kappa in kappas:
        errors_by_layer = [[] for _ in range(num_layers)]
        oracle_errors = []

        for _ in range(n_test // 50):
            bs = 50
            seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
                bs, n_support, p_dim, kappa, cfg['noise'], device)

            with torch.no_grad():
                _, intermediates = model.forward_with_intermediates(seq)

            # Oracle: ridge regression on dot-product kernel
            x_s_np = x_s.cpu().numpy()
            y_s_np = y_s.cpu().numpy()
            x_q_np = x_q.squeeze(1).cpu().numpy()
            y_q_np = y_q.cpu().numpy()

            for i in range(bs):
                K = x_s_np[i] @ x_s_np[i].T
                A = K + lam * np.eye(n_support)
                try:
                    alpha_star = np.linalg.solve(A, y_s_np[i])
                except np.linalg.LinAlgError:
                    continue
                k_q = x_s_np[i] @ x_q_np[i]
                f_oracle = k_q @ alpha_star

                for l in range(num_layers):
                    h_l = intermediates[l + 1][i:i+1]
                    pred_l = readouts[l](h_l).item()
                    errors_by_layer[l].append((pred_l - f_oracle) ** 2)
                oracle_errors.append((y_q_np[i] - f_oracle) ** 2)

        # Mean error per layer
        mean_errs = [np.mean(e) if e else float('nan') for e in errors_by_layer]
        # Normalize by initial error (layer 0 error)
        if mean_errs[0] > 1e-12:
            norm_errs = [e / mean_errs[0] for e in mean_errs]
        else:
            norm_errs = mean_errs

        # Theoretical rates
        rho_cg = ((math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)) ** 2
        rho_gd = ((kappa - 1) / (kappa + 1)) ** 2
        cg_theory = [rho_cg ** (l + 1) for l in range(num_layers)]
        gd_theory = [rho_gd ** (l + 1) for l in range(num_layers)]

        results[kappa] = {
            'mean_errors': mean_errs,
            'normalized_errors': norm_errs,
            'cg_theory': cg_theory,
            'gd_theory': gd_theory,
        }

    return results

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(losses, out_dir):
    """Plot training loss curve."""
    plt.figure(figsize=(8, 4))
    window = min(500, len(losses) // 10 + 1)
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.semilogy(smoothed)
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')
    plt.title('ICL Transformer Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/training_loss.png', dpi=150)
    plt.close()


def plot_probes(cg_sims, gd_sims, ctrl_sims, out_dir):
    """Plot probe cosine similarity by layer."""
    layers = list(range(1, len(cg_sims) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(layers, cg_sims, '-o', color='green', label='CG probe', linewidth=2)
    plt.plot(layers, gd_sims, '--s', color='red', label='GD probe', linewidth=2)
    plt.plot(layers, ctrl_sims, ':x', color='gray', label='Random control', linewidth=1)
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity (test set)')
    plt.title('Linear Probe Recovery: CG vs GD States from Trained Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.15, 1.05)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/probe_cosine_sim.png', dpi=150)
    plt.close()


def plot_convergence(conv_results, out_dir):
    """Plot per-layer convergence rates vs CG/GD theory."""
    fig, axes = plt.subplots(1, len(conv_results), figsize=(5*len(conv_results), 4),
                             squeeze=False)
    for i, (kappa, data) in enumerate(sorted(conv_results.items())):
        ax = axes[0, i]
        layers = list(range(1, len(data['normalized_errors']) + 1))
        ax.semilogy(layers, data['normalized_errors'], '-o', color='blue',
                     label='Trained model', markersize=4)
        ax.semilogy(layers, data['cg_theory'], '--', color='green',
                     label='CG theory', linewidth=1.5)
        ax.semilogy(layers, data['gd_theory'], ':', color='red',
                     label='GD theory', linewidth=1.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Normalized Error')
        ax.set_title(f'kappa = {kappa:.0f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-8)
    plt.suptitle('Per-Layer Convergence: Trained Model vs CG/GD Theory', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/convergence_rates.png', dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train ICL transformer and probe for CG structure')
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
    parser.add_argument('--out-dir', type=str, default='docs/figures/trained')
    parser.add_argument('--n-probe-problems', type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    # ---- Phase 1: Train ----
    print(f"\n{'='*60}")
    print("Phase 1: Training ICL Transformer")
    print(f"  {args.num_layers} layers, d={args.d_model}, {args.nhead} heads")
    print(f"  {args.steps} steps, batch={args.batch_size}")
    print(f"  n_support={args.n_support}, p={args.p}, noise={args.noise}")
    print(f"{'='*60}")

    model = ICLTransformer(
        input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, max_seq_len=args.n_support + 1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    losses = train_model(model, cfg, device)
    train_time = time.time() - t0
    print(f"\n  Training complete in {train_time/60:.1f} minutes")
    print(f"  Final loss (avg last 1000): {np.mean(losses[-1000:]):.5f}")

    plot_training(losses, args.out_dir)

    # Save model
    ckpt_path = f'{args.out_dir}/model.pt'
    torch.save({'model': model.state_dict(), 'cfg': cfg, 'args': vars(args)}, ckpt_path)
    print(f"  Saved model to {ckpt_path}")

    # ---- Phase 2: Train per-layer readouts ----
    print(f"\n{'='*60}")
    print("Phase 2: Training per-layer readout heads")
    print(f"{'='*60}")
    t0 = time.time()
    readouts = train_layer_readouts(model, cfg, device, readout_steps=3000)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Phase 3: Probe for CG / GD states ----
    print(f"\n{'='*60}")
    print("Phase 3: Probing for CG vs GD states")
    print(f"  Collecting {args.n_probe_problems} test problems...")
    print(f"{'='*60}")
    t0 = time.time()
    all_acts, all_cg, all_gd = collect_probe_data(
        model, cfg, device, n_problems=args.n_probe_problems)
    print(f"  Data collection: {time.time()-t0:.1f}s")

    t0 = time.time()
    cg_sims, gd_sims, ctrl_sims = run_probe_analysis(
        all_acts, all_cg, all_gd, args.num_layers)
    print(f"  Probe fitting: {time.time()-t0:.1f}s")

    print("\n  Layer | CG sim | GD sim | Random")
    print("  " + "-" * 40)
    for l in range(args.num_layers):
        print(f"  {l+1:5d} | {cg_sims[l]:6.3f} | {gd_sims[l]:6.3f} | {ctrl_sims[l]:6.3f}")

    # Which is the model closer to?
    cg_mean = np.mean(cg_sims)
    gd_mean = np.mean(gd_sims)
    print(f"\n  Mean CG probe similarity: {cg_mean:.3f}")
    print(f"  Mean GD probe similarity: {gd_mean:.3f}")
    if cg_mean > gd_mean + 0.02:
        print("  => Model internal states are MORE aligned with CG than GD")
    elif gd_mean > cg_mean + 0.02:
        print("  => Model internal states are MORE aligned with GD than CG")
    else:
        print("  => CG and GD probe similarities are comparable")

    plot_probes(cg_sims, gd_sims, ctrl_sims, args.out_dir)

    # ---- Phase 4: Convergence rate analysis ----
    print(f"\n{'='*60}")
    print("Phase 4: Convergence rate analysis")
    print(f"{'='*60}")
    t0 = time.time()
    conv_results = run_convergence_analysis(
        model, readouts, cfg, device, kappas=[1.0, 5.0, 10.0, 50.0])
    print(f"  Done in {time.time()-t0:.1f}s")

    for kappa, data in sorted(conv_results.items()):
        print(f"\n  kappa={kappa:.0f}:")
        print(f"    Model error (layer 1->last): "
              f"{data['normalized_errors'][0]:.4f} -> {data['normalized_errors'][-1]:.6f}")
        print(f"    CG theory  (layer 1->last): "
              f"{data['cg_theory'][0]:.4f} -> {data['cg_theory'][-1]:.6f}")
        print(f"    GD theory  (layer 1->last): "
              f"{data['gd_theory'][0]:.4f} -> {data['gd_theory'][-1]:.6f}")

    plot_convergence(conv_results, args.out_dir)

    # ---- Save results ----
    results = {
        'training': {
            'final_loss': float(np.mean(losses[-1000:])),
            'train_time_min': train_time / 60,
            'n_params': n_params,
        },
        'probes': {
            'cg_sims': cg_sims, 'gd_sims': gd_sims, 'ctrl_sims': ctrl_sims,
            'cg_mean': float(cg_mean), 'gd_mean': float(gd_mean),
        },
        'convergence': {
            str(k): {
                'normalized_errors': v['normalized_errors'],
                'cg_theory': v['cg_theory'],
                'gd_theory': v['gd_theory'],
            } for k, v in conv_results.items()
        },
        'config': cfg,
    }
    with open(f'{args.out_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out_dir}/results.json")
    print(f"  Figures saved to {args.out_dir}/")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Trained {args.num_layers}-layer transformer ({n_params:,} params) "
          f"on ICL regression in {train_time/60:.1f} min")
    print(f"  Final training loss: {np.mean(losses[-1000:]):.5f}")
    print(f"  CG probe mean cosine sim: {cg_mean:.3f}")
    print(f"  GD probe mean cosine sim: {gd_mean:.3f}")
    if cg_mean > gd_mean + 0.02:
        print(f"  FINDING: Trained model's internal states are more CG-like than GD-like")
    elif gd_mean > cg_mean + 0.02:
        print(f"  FINDING: Trained model's internal states are more GD-like than CG-like")
    else:
        print(f"  FINDING: CG and GD probes are comparably aligned")
    print(f"\n  See {args.out_dir}/ for figures and detailed results.")


if __name__ == '__main__':
    main()
