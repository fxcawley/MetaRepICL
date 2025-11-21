import os
import numpy as np
import torch

def save_probe_weights(weights: np.ndarray, path: str):
    """Save probe weights to npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, weights)

def visualize_probe_cosine_sim(
    cosine_sims: np.ndarray, 
    layer_indices: List[int], 
    save_path: str
):
    """
    Plot cosine similarity of probe recovery across layers.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.plot(layer_indices, cosine_sims, 'o-')
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Probe Recovery of CG State")
    plt.grid(True, alpha=0.3)
    plt.axhline(0.9, color='g', linestyle='--', label="Success Threshold (0.9)")
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

