from typing import List, Dict
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_ci(values: List[float], alpha: float = 0.05) -> Dict[str, float]:
	x = np.array(values, dtype=np.float64)
	m = float(np.mean(x))
	s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
	n = len(x)
	# normal approx for brevity; replace with t-dist if needed
	z = 1.96 if alpha == 0.05 else 1.96
	ci = z * (s / np.sqrt(max(n, 1))) if n > 1 else 0.0
	return {"mean": m, "ci": ci, "n": n}
