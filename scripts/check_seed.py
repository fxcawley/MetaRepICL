#!/usr/bin/env python3
"""Pre-commit hook: verify experiment scripts accept a seed parameter.

Scans experiment Python files to ensure they either:
  1. Use @hydra.main (seed comes from config), or
  2. Accept a --seed / seed= argument, or
  3. Call set_seed() or np.random.default_rng(seed).

Exits non-zero if a file appears to use randomness without seed control.
"""

import sys
from pathlib import Path


SEED_PATTERNS = {
    "hydra.main",
    "set_seed",
    "default_rng",
    "manual_seed",
    "random.seed",
    "np.random.seed",
    "torch.manual_seed",
    "seed=",
    "--seed",
}


def check_file(path: Path) -> bool:
    """Return True if the file has seed control, False if suspicious."""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return True  # Can't read, skip

    # Skip non-runnable files (pure libraries, __init__, etc.)
    if path.name.startswith("__"):
        return True

    has_randomness = any(
        kw in source
        for kw in ["random", "np.random", "torch.manual_seed", "rng"]
    )
    if not has_randomness:
        return True  # No randomness, no seed needed

    has_seed = any(pattern in source for pattern in SEED_PATTERNS)
    return has_seed


def main() -> int:
    files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]
    failures = []

    for f in files:
        if not check_file(f):
            failures.append(str(f))

    if failures:
        print("Seed enforcement FAILED for:")
        for f in failures:
            print(f"  {f}")
        print(
            "\nExperiment scripts must have explicit seed control "
            "(hydra config, --seed arg, or set_seed() call)."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
