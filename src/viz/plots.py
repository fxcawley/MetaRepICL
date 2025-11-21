import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import os

def plot_attention_maps(
    attention_weights: np.ndarray,
    save_path: str,
    title: str = "Attention Weights",
    xlabel: str = "Key Token Index",
    ylabel: str = "Query Token Index"
):
    """
    Plots a heatmap of attention weights.
    
    Args:
        attention_weights: 2D array (n_queries, n_keys)
        save_path: path to save the figure
        title: plot title
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_cg_trajectory(
    errors: List[float],
    save_path: str,
    labels: Optional[List[str]] = None,
    title: str = "CG Convergence",
    log_scale: bool = True
):
    """
    Plots convergence trajectory of CG or ICL steps.
    
    Args:
        errors: List of error values per step (or list of lists for multiple runs)
        save_path: path to save
        labels: labels for multiple runs (optional)
    """
    plt.figure(figsize=(8, 6))
    
    # Check if list of lists
    if errors and isinstance(errors[0], list):
        for i, err_seq in enumerate(errors):
            lbl = labels[i] if labels and i < len(labels) else f"Run {i+1}"
            plt.plot(err_seq, marker='o', linestyle='-', label=lbl)
    else:
        plt.plot(errors, marker='o', linestyle='-', label="Error")

    if log_scale:
        plt.yscale('log')
        
    plt.xlabel("Iteration / Layer")
    plt.ylabel("Error (Residual Norm)")
    plt.title(title)
    if labels or (errors and isinstance(errors[0], list)):
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def overlay_cg_rate(
    ax,
    kappa: float,
    steps: int,
    color: str = 'red',
    linestyle: str = '--'
):
    """
    Overlays the theoretical CG convergence rate onto an existing axis.
    Rate: ((sqrt(k)-1)/(sqrt(k)+1))^t
    """
    rate = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    t = np.arange(steps)
    # Assuming starting error is normalized to 1
    y = rate ** t
    ax.plot(t, y, color=color, linestyle=linestyle, label=f"Theory (κ={kappa})")

