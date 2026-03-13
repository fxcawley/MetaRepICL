"""Logging utilities for MetaRep experiments."""

import os
import sys
import hashlib
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def get_env_info() -> Dict[str, str]:
    """Capture environment info for reproducibility."""
    info = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info


def config_hash(cfg: DictConfig) -> str:
    """Compute a deterministic hash of the config for fingerprinting."""
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(yaml_str.encode()).hexdigest()[:12]


def init_logging(cfg: DictConfig, project: str = "metarep") -> Optional[Any]:
    """Initialize W&B logging if enabled in config.

    Returns the wandb run object if enabled, None otherwise.
    """
    log_cfg = OmegaConf.to_container(cfg, resolve=True)
    env_info = get_env_info()
    cfg_hash = config_hash(cfg)

    wandb_cfg = cfg.get("wandb", {})
    enabled = wandb_cfg.get("enabled", False)

    if not enabled:
        print(f"[logging] W&B disabled. Config hash: {cfg_hash}")
        print(f"[logging] Env: Python {env_info['python'].split()[0]}, "
              f"torch {env_info['torch']}, git {env_info['git_commit'][:8]}")
        return None

    try:
        import wandb
        mode = wandb_cfg.get("mode", "offline")
        run = wandb.init(
            project=wandb_cfg.get("project", project),
            config=log_cfg,
            mode=mode,
            tags=[f"hash:{cfg_hash}"],
        )
        wandb.config.update({"env": env_info})
        print(f"[logging] W&B initialized ({mode}). Config hash: {cfg_hash}")
        return run
    except Exception as e:
        print(f"[logging] W&B init failed: {e}. Continuing without W&B.")
        return None


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to W&B if active, otherwise print."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
            return
    except Exception:
        pass
    print(f"[metrics] step={step} {metrics}")
