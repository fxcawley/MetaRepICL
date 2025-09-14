import numpy as np
from omegaconf import DictConfig
import hydra
from src.data.synth_linear import make_linear_dataset, krr_oracle


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	Xs, ys, Xq, yq = make_linear_dataset(
		n_support=cfg.get("n_support", 32),
		n_query=cfg.get("n_query", 32),
		p=cfg.get("p", 16),
		noise=cfg.get("noise", 0.1),
		seed=cfg.seed,
	)
	lam = float(cfg.get("lambda", 1e-2))
	pred = krr_oracle(Xs, ys, Xq, lam)
	rmse = float(np.sqrt(np.mean((pred - yq) ** 2)))
	print({"rmse": rmse, "lambda": lam, "seed": int(cfg.seed)})


if __name__ == "__main__":
	main()
