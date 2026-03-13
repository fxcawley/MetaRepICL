"""Mixed-kappa training: train on ICL tasks with varying condition numbers.

Trains a transformer on tasks with kappa in {1, 10, 50, 100, 500},
then probes and measures convergence rates STRATIFIED by kappa.
This is the fairest test of CG vs GD: at high kappa, CG and GD rates
diverge maximally, and CG state variables retain variance across problems.

Usage:
    python experiments/train_mixed_kappa.py                    # full run
    python experiments/train_mixed_kappa.py --steps 5000       # quick test
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
# Data generation
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
    """Generate batch with a randomly chosen kappa from the list."""
    kappa = kappas[torch.randint(0, len(kappas), (1,)).item()]
    seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
        batch_size, n_support, p, kappa, noise, device)
    return seq, y_q

# ---------------------------------------------------------------------------
# CG / GD reference trajectories
# ---------------------------------------------------------------------------

def cg_trajectory(K, y, lam, steps):
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

# ---------------------------------------------------------------------------
# Train per-layer readout heads
# ---------------------------------------------------------------------------

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
# Probing stratified by kappa
# ---------------------------------------------------------------------------

def fit_probe(X_train, y_train, X_test, y_test, lam_reg=1e-3):
    XtX = X_train.T @ X_train
    reg = lam_reg * np.eye(XtX.shape[0])
    try:
        W = np.linalg.solve(XtX + reg, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        return 0.0
    pred = X_test @ W
    p_flat, y_flat = pred.flatten(), y_test.flatten()
    norms = np.linalg.norm(p_flat) * np.linalg.norm(y_flat)
    if norms < 1e-12:
        return 0.0
    return float(np.dot(p_flat, y_flat) / norms)


def run_stratified_analysis(model, readouts, cfg, device, n_per_kappa=200):
    """Probe and measure convergence, stratified by kappa.

    CORRECTED: Convergence comparison uses actual CG/GD prediction errors
    at each step, not theoretical rate bounds (which used the wrong kappa --
    the feature covariance condition number, not cond(K+lambda*I)).
    """
    model.eval()
    num_layers = model.num_layers
    n_support = cfg['n_support']
    p_dim = cfg['p']
    lam = cfg.get('probe_lam', 0.1)

    results = {}
    for kappa in KAPPAS:
        print(f"  Analyzing kappa={kappa:.0f}...")
        # Collect data for this kappa
        all_acts = {l: [] for l in range(num_layers + 1)}
        all_cg = {l: [] for l in range(1, num_layers + 1)}
        all_gd = {l: [] for l in range(1, num_layers + 1)}
        per_layer_model_errors = [[] for _ in range(num_layers)]
        per_layer_cg_errors = [[] for _ in range(num_layers)]
        per_layer_gd_errors = [[] for _ in range(num_layers)]
        actual_kappa_Ks = []

        bs = 50
        for start in range(0, n_per_kappa, bs):
            cur_bs = min(bs, n_per_kappa - start)
            seq, y_q, x_s, y_s, x_q = generate_batch_with_cov(
                cur_bs, n_support, p_dim, kappa, cfg['noise'], device)

            with torch.no_grad():
                _, intermediates = model.forward_with_intermediates(seq)

            for l in range(num_layers + 1):
                all_acts[l].append(intermediates[l].cpu().numpy())

            x_s_np = x_s.cpu().numpy()
            y_s_np = y_s.cpu().numpy()
            x_q_np = x_q.squeeze(1).cpu().numpy()

            for i in range(cur_bs):
                K = x_s_np[i] @ x_s_np[i].T
                y_vec = y_s_np[i]
                A = K + lam * np.eye(n_support)
                try:
                    alpha_star = np.linalg.solve(A, y_vec)
                except np.linalg.LinAlgError:
                    continue
                k_q = x_s_np[i] @ x_q_np[i]
                f_oracle = float(k_q @ alpha_star)

                # Track actual cond(K + lambda I) for diagnostics
                eigs = np.linalg.eigvalsh(A)
                actual_kappa_Ks.append(float(eigs[-1] / max(eigs[0], 1e-30)))

                # Compute actual CG and GD prediction errors at each step
                cg_traj = cg_trajectory(K, y_vec, lam, num_layers)
                gd_traj = gd_trajectory(K, y_vec, lam, num_layers)
                for l in range(1, num_layers + 1):
                    all_cg[l].append(cg_traj[l])
                    all_gd[l].append(gd_traj[l])

                for l in range(num_layers):
                    # Model prediction error at layer l
                    h_l = intermediates[l + 1][i:i+1]
                    with torch.no_grad():
                        pred_l = readouts[l](h_l).item()
                    per_layer_model_errors[l].append((pred_l - f_oracle) ** 2)

                    # CG prediction error at step l+1
                    alpha_cg_l = cg_traj[l + 1]
                    f_cg_l = float(k_q @ alpha_cg_l)
                    per_layer_cg_errors[l].append((f_cg_l - f_oracle) ** 2)

                    # GD prediction error at step l+1
                    alpha_gd_l = gd_traj[l + 1]
                    f_gd_l = float(k_q @ alpha_gd_l)
                    per_layer_gd_errors[l].append((f_gd_l - f_oracle) ** 2)

        # Stack arrays
        for l in range(num_layers + 1):
            all_acts[l] = np.concatenate(all_acts[l], axis=0)
        for l in range(1, num_layers + 1):
            all_cg[l] = np.array(all_cg[l])
            all_gd[l] = np.array(all_gd[l])

        # Fit probes
        n = all_acts[0].shape[0]
        split = int(0.8 * n)
        cg_sims, gd_sims, ctrl_sims = [], [], []
        for l in range(1, num_layers + 1):
            X = all_acts[l]
            X_tr, X_te = X[:split], X[split:]
            cg_tr, cg_te = all_cg[l][:split], all_cg[l][split:]
            gd_tr, gd_te = all_gd[l][:split], all_gd[l][split:]
            rng = np.random.default_rng(42 + l)
            ctrl = rng.standard_normal(all_cg[l].shape)
            ctrl_tr, ctrl_te = ctrl[:split], ctrl[split:]
            cg_sims.append(fit_probe(X_tr, cg_tr, X_te, cg_te))
            gd_sims.append(fit_probe(X_tr, gd_tr, X_te, gd_te))
            ctrl_sims.append(fit_probe(X_tr, ctrl_tr, X_te, ctrl_te))

        # Convergence: actual CG/GD prediction errors (not theoretical bounds)
        model_errs = [np.mean(e) if e else float('nan') for e in per_layer_model_errors]
        cg_errs = [np.mean(e) if e else float('nan') for e in per_layer_cg_errors]
        gd_errs = [np.mean(e) if e else float('nan') for e in per_layer_gd_errors]
        # Normalize all by GD step-1 error for a common baseline
        baseline = gd_errs[0] if gd_errs[0] > 1e-12 else 1.0
        model_norm = [e / baseline for e in model_errs]
        cg_norm = [e / baseline for e in cg_errs]
        gd_norm = [e / baseline for e in gd_errs]

        mean_actual_kappa = float(np.mean(actual_kappa_Ks)) if actual_kappa_Ks else float('nan')

        results[kappa] = {
            'cg_sims': cg_sims, 'gd_sims': gd_sims, 'ctrl_sims': ctrl_sims,
            'cg_mean': float(np.mean(cg_sims)),
            'gd_mean': float(np.mean(gd_sims)),
            'model_norm': model_norm,
            'cg_norm': cg_norm,
            'gd_norm': gd_norm,
            'model_raw': model_errs,
            'cg_raw': cg_errs,
            'gd_raw': gd_errs,
            'actual_kappa_K': mean_actual_kappa,
        }

    return results

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(losses, out_dir):
    plt.figure(figsize=(8, 4))
    w = min(500, len(losses) // 10 + 1)
    plt.semilogy(np.convolve(losses, np.ones(w)/w, mode='valid'))
    plt.xlabel('Step'); plt.ylabel('MSE Loss')
    plt.title('Mixed-Kappa ICL Training Loss')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f'{out_dir}/training_loss_mixed.png', dpi=150); plt.close()


def plot_stratified_probes(results, out_dir):
    """Plot probe cosine sim by layer, one panel per kappa."""
    kappas = sorted(results.keys())
    fig, axes = plt.subplots(1, len(kappas), figsize=(4*len(kappas), 4), squeeze=False)
    for i, kappa in enumerate(kappas):
        ax = axes[0, i]
        d = results[kappa]
        layers = list(range(1, len(d['cg_sims']) + 1))
        ax.plot(layers, d['cg_sims'], '-o', color='green', markersize=3, label='CG')
        ax.plot(layers, d['gd_sims'], '--s', color='red', markersize=3, label='GD')
        ax.plot(layers, d['ctrl_sims'], ':x', color='gray', markersize=3, label='Random')
        ax.set_xlabel('Layer'); ax.set_ylabel('Cosine Sim')
        ax.set_title(f'kappa={kappa:.0f}\nCG={d["cg_mean"]:.3f} GD={d["gd_mean"]:.3f}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.15, 1.05)
    plt.suptitle('CG vs GD Probe Recovery (Mixed-Kappa Trained Model)', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/probe_by_kappa.png', dpi=150, bbox_inches='tight'); plt.close()


def plot_stratified_convergence(results, out_dir):
    """Plot per-layer convergence: model vs actual CG vs actual GD."""
    kappas = sorted(results.keys())
    fig, axes = plt.subplots(1, len(kappas), figsize=(4*len(kappas), 4), squeeze=False)
    for i, kappa in enumerate(kappas):
        ax = axes[0, i]
        d = results[kappa]
        layers = list(range(1, len(d['model_norm']) + 1))
        ax.semilogy(layers, d['model_norm'], '-o', color='blue',
                     markersize=4, label='Model')
        ax.semilogy(layers, d['cg_norm'], '--', color='green', lw=1.5, label='Actual CG')
        ax.semilogy(layers, d['gd_norm'], ':', color='red', lw=1.5, label='Actual GD')
        ax.set_xlabel('Layer / Step'); ax.set_ylabel('Norm. Pred. Error')
        actual_k = d.get('actual_kappa_K', kappa)
        ax.set_title(f'kappa_input={kappa:.0f}\ncond(K+lI)={actual_k:.0f}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-10)
    plt.suptitle('Per-Layer Convergence: Model vs Actual CG/GD Trajectories', y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/convergence_by_kappa.png', dpi=150, bbox_inches='tight'); plt.close()


def plot_summary(results, out_dir):
    """Summary: CG vs GD probe mean across kappas."""
    kappas = sorted(results.keys())
    cg_means = [results[k]['cg_mean'] for k in kappas]
    gd_means = [results[k]['gd_mean'] for k in kappas]
    x = np.arange(len(kappas))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, cg_means, w, label='CG probe', color='green', alpha=0.8)
    plt.bar(x + w/2, gd_means, w, label='GD probe', color='red', alpha=0.8)
    plt.xticks(x, [f'{k:.0f}' for k in kappas])
    plt.xlabel('Condition Number (kappa)')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('CG vs GD Probe Similarity by Condition Number')
    plt.legend(); plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(f'{out_dir}/cg_vs_gd_by_kappa.png', dpi=150); plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--out-dir', type=str, default='docs/figures/trained_mixed')
    parser.add_argument('--n-per-kappa', type=int, default=200)
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

    # ---- Phase 1: Train on mixed kappa ----
    print(f"\n{'='*60}")
    print(f"Phase 1: Training on mixed kappa {KAPPAS}")
    print(f"  {args.num_layers} layers, d={args.d_model}, {args.steps} steps")
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
    print(f"\n  Training: {train_time/60:.1f} min, final loss: {np.mean(losses[-1000:]):.5f}")
    plot_training(losses, args.out_dir)
    torch.save({'model': model.state_dict(), 'cfg': cfg}, f'{args.out_dir}/model_mixed.pt')

    # ---- Phase 2: Train per-layer readouts ----
    print(f"\n{'='*60}")
    print("Phase 2: Training per-layer readout heads")
    print(f"{'='*60}")
    t0 = time.time()
    readouts = train_layer_readouts(model, cfg, device, readout_steps=3000)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Phase 3: Stratified probe + convergence analysis ----
    print(f"\n{'='*60}")
    print("Phase 3: Stratified analysis by kappa")
    print(f"{'='*60}")
    t0 = time.time()
    results = run_stratified_analysis(model, readouts, cfg, device, args.n_per_kappa)
    print(f"  Total analysis time: {time.time()-t0:.1f}s")

    # Print summary table
    print(f"\n  Probes:")
    print(f"  {'kappa':>6} | {'CG sim':>7} | {'GD sim':>7} | {'Winner':>6}")
    print("  " + "-" * 40)
    for kappa in sorted(results.keys()):
        d = results[kappa]
        winner = "CG" if d['cg_mean'] > d['gd_mean'] + 0.02 else \
                 "GD" if d['gd_mean'] > d['cg_mean'] + 0.02 else "tie"
        print(f"  {kappa:6.0f} | {d['cg_mean']:7.3f} | {d['gd_mean']:7.3f} | {winner:>6}")

    print(f"\n  Convergence (normalized pred error at layer 12):")
    print(f"  {'kappa':>6} | {'cond(K+lI)':>10} | {'Model':>8} | {'CG':>8} | {'GD':>8} | Verdict")
    print("  " + "-" * 65)
    for kappa in sorted(results.keys()):
        d = results[kappa]
        m = d['model_norm'][-1]
        c = d['cg_norm'][-1]
        g = d['gd_norm'][-1]
        ak = d.get('actual_kappa_K', kappa)
        if m < c * 0.5 and m < g * 0.5:
            v = "Model > CG > GD"
        elif m < g * 0.5:
            v = "Model ~ CG >> GD"
        elif m > g:
            v = "Model > GD"
        else:
            v = "comparable"
        print(f"  {kappa:6.0f} | {ak:10.0f} | {m:8.4f} | {c:8.4f} | {g:8.4f} | {v}")

    # ---- Phase 4: Plots ----
    plot_stratified_probes(results, args.out_dir)
    plot_stratified_convergence(results, args.out_dir)
    plot_summary(results, args.out_dir)

    # ---- Save ----
    save_results = {
        'training': {'final_loss': float(np.mean(losses[-1000:])),
                     'train_time_min': train_time / 60, 'n_params': n_params},
        'kappas_trained_on': KAPPAS,
    }
    for kappa in sorted(results.keys()):
        d = results[kappa]
        save_results[f'kappa_{kappa:.0f}'] = {
            'cg_sims': d['cg_sims'], 'gd_sims': d['gd_sims'],
            'cg_mean': d['cg_mean'], 'gd_mean': d['gd_mean'],
            'model_norm': d['model_norm'],
            'cg_norm': d['cg_norm'],
            'gd_norm': d['gd_norm'],
            'actual_kappa_K': d['actual_kappa_K'],
        }
    with open(f'{args.out_dir}/results_mixed.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n  Results saved to {args.out_dir}/")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Trained on kappas {KAPPAS} for {args.steps} steps ({train_time/60:.1f} min)")

    # Overall verdict
    all_cg = [results[k]['cg_mean'] for k in sorted(results.keys())]
    all_gd = [results[k]['gd_mean'] for k in sorted(results.keys())]
    if np.mean(all_cg) > np.mean(all_gd) + 0.02:
        print("  PROBES: Model internal states are more CG-like across kappas")
    elif np.mean(all_gd) > np.mean(all_cg) + 0.02:
        print("  PROBES: Model internal states are more GD-like across kappas")
    else:
        print("  PROBES: CG and GD probes are comparable across kappas")

    # Convergence verdict
    for kappa in sorted(results.keys()):
        d = results[kappa]
        m, c, g = d['model_norm'][-1], d['cg_norm'][-1], d['gd_norm'][-1]
        if m < g * 0.5:
            print(f"  CONVERGENCE (kappa={kappa:.0f}): Model >> GD "
                  f"(model={m:.4f}, CG={c:.4f}, GD={g:.4f})")


if __name__ == '__main__':
    main()
