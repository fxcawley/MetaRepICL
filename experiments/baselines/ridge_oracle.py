import numpy as np
from omegaconf import DictConfig
import hydra

from src.data.synth_linear import make_linear_dataset, krr_oracle


def run_ridge_oracle(
	n_support: int = 32,
	n_query: int = 32,
	p: int = 16,
	noise: float = 0.1,
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
	pred = krr_oracle(Xs, ys, Xq, lam)
	rmse = float(np.sqrt(np.mean((pred - yq) ** 2)))
	return {"rmse": rmse, "lambda": lam, "seed": seed}


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	res = run_ridge_oracle(
		n_support=cfg.get("n_support", 32),
		n_query=cfg.get("n_query", 32),
		p=cfg.get("p", 16),
		noise=cfg.get("noise", 0.1),
		lam=float(cfg.get("lambda", 1e-2)),
		seed=int(cfg.seed),
	)
	print(res)


if __name__ == "__main__":
	main()
