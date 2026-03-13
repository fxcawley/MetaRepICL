from typing import List, Dict
import math
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_ci(values: List[float], alpha: float = 0.05, use_t: bool = True) -> Dict[str, float]:
	x = np.array(values, dtype=np.float64)
	m = float(np.mean(x))
	n = len(x)
	if n <= 1:
		return {"mean": m, "ci": 0.0, "n": n}
	s = float(np.std(x, ddof=1))
	if use_t:
		try:
			from scipy import stats
			t_val = float(stats.t.ppf(1 - alpha / 2.0, df=n - 1))
			scale = t_val
		except ImportError:
			raise RuntimeError("scipy is required for use_t=True in mean_ci")
	else:
		if alpha == 0.05:
			scale = 1.96
		else:
			try:
				from scipy.stats import norm
				scale = float(norm.ppf(1 - alpha / 2.0))
			except ImportError:
				raise RuntimeError("scipy is required for non-0.05 alpha when use_t=False")
	ci = scale * (s / math.sqrt(n))
	return {"mean": m, "ci": float(ci), "n": n}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Classification accuracy (for binary/multiclass predictions)."""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	if y_pred.dtype in (np.float32, np.float64):
		y_pred = (y_pred >= 0.5).astype(int)
	return float(np.mean(y_true == y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Coefficient of determination (R^2)."""
	y_true = np.asarray(y_true, dtype=np.float64)
	y_pred = np.asarray(y_pred, dtype=np.float64)
	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	if ss_tot < 1e-12:
		return 1.0 if ss_res < 1e-12 else 0.0
	return float(1.0 - ss_res / ss_tot)
