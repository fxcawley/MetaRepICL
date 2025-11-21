import numpy as np
from omegaconf import DictConfig
import hydra

from src.data.synth_linear import make_linear_dataset


def gd_linear_fit(Xs: np.ndarray, ys: np.ndarray, steps: int, lr: float, lam: float, seed: int) -> np.ndarray:
	"""Gradient descent with L2 regularization on a linear predictor w."""
	rng = np.random.default_rng(seed)
	p = Xs.shape[1]
	w = rng.standard_normal(size=(p,)).astype(np.float64) * 0.01
	for _ in range(steps):
		# grad = X^T (X w - y) + lam * w
		grad = Xs.T @ (Xs @ w - ys) + lam * w
		w -= lr * grad / Xs.shape[0]
	return w


def run_gd_icl(
	n_support: int = 32,
	n_query: int = 32,
	p: int = 16,
	noise: float = 0.1,
	steps: int = 200,
	lr: float = 0.1,
	lam: float = 1e-2,
	seed: int = 123,
) -> dict:
	Xs, ys, Xq, yq = make_linear_dataset(
		n_support=n_support,
		n_query=n_query,
		p=p,
		noise=noise,
		seed=seed,
	)
	w = gd_linear_fit(Xs, ys, steps=steps, lr=lr, lam=lam, seed=seed)
	pred = Xq @ w
	rmse = float(np.sqrt(np.mean((pred - yq) ** 2)))
	return {"rmse": rmse, "steps": steps, "lr": lr, "lambda": lam, "seed": seed}


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	res = run_gd_icl(
		n_support=cfg.get("n_support", 32),
		n_query=cfg.get("n_query", 32),
		p=cfg.get("p", 16),
		noise=cfg.get("noise", 0.1),
		steps=int(cfg.get("steps", 200)),
		lr=float(cfg.get("lr", 0.1)),
		lam=float(cfg.get("lambda", 1e-2)),
		seed=int(cfg.seed),
	)
	print(res)


if __name__ == "__main__":
	main()
