"""Checkpoint save/load utilities for MetaRep experiments.

Supports saving and resuming training state including model weights,
optimizer state, step count, metrics, and RNG states.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def save_checkpoint(
    path: str,
    step: int,
    model_state: Optional[Dict[str, Any]] = None,
    optimizer_state: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a training checkpoint.

    Args:
        path: Directory to save checkpoint in.
        step: Current training step.
        model_state: Model state dict (from model.state_dict()).
        optimizer_state: Optimizer state dict.
        metrics: Current metrics dict.
        config: Config snapshot for reproducibility.
        rng_state: RNG states for exact resumption.

    Returns:
        Path to saved checkpoint file.
    """
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, f"checkpoint_step{step}.pt")

    ckpt = {
        "step": step,
        "metrics": metrics or {},
        "config": config or {},
    }

    if model_state is not None:
        ckpt["model_state_dict"] = model_state
    if optimizer_state is not None:
        ckpt["optimizer_state_dict"] = optimizer_state

    # Capture RNG states for exact resumption
    if rng_state is None:
        rng_state = {
            "python_rng": None,
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda_rng"] = torch.cuda.get_rng_state_all()
    ckpt["rng_state"] = rng_state

    torch.save(ckpt, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint.

    Args:
        path: Path to checkpoint file.
        map_location: Device to map tensors to.

    Returns:
        Checkpoint dict with keys: step, model_state_dict, optimizer_state_dict,
        metrics, config, rng_state.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt


def restore_rng_state(rng_state: Dict[str, Any]) -> None:
    """Restore RNG states from a checkpoint for exact resumption."""
    if "numpy_rng" in rng_state and rng_state["numpy_rng"] is not None:
        np.random.set_state(rng_state["numpy_rng"])
    if "torch_rng" in rng_state and rng_state["torch_rng"] is not None:
        torch.random.set_rng_state(rng_state["torch_rng"])
    if "cuda_rng" in rng_state and rng_state.get("cuda_rng") is not None:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda_rng"])


def find_latest_checkpoint(path: str) -> Optional[str]:
    """Find the latest checkpoint in a directory by step number.

    Returns:
        Path to latest checkpoint file, or None if no checkpoints found.
    """
    ckpt_dir = Path(path)
    if not ckpt_dir.exists():
        return None

    ckpts = sorted(
        ckpt_dir.glob("checkpoint_step*.pt"),
        key=lambda p: int(p.stem.split("step")[1]),
    )
    if not ckpts:
        return None
    return str(ckpts[-1])
