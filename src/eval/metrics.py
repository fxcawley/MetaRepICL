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
